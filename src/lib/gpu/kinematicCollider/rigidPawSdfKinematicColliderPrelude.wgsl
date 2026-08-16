override KINEMATIC_COLLIDER_COUNT: u32 = 0u;
override KINEMATIC_COLLIDER_SDF_RESOLUTION: u32 = 2u;

const KINEMATIC_COLLIDER_EPSILON: f32 = 1e-7;
const KINEMATIC_COLLIDER_NO_CONTACT_DISTANCE: f32 = 1e30;

struct KinematicCollider {
    currentWorldFromUnit: mat4x4f,
    currentUnitFromWorld: mat4x4f,
    previousWorldFromUnit: mat4x4f,
    timing: vec4f,
}

struct KinematicColliderFieldSample {
    signedDistance: f32,
    gradient: vec3f,
    valid: bool,
}

struct KinematicColliderSample {
    signedDistance: f32,
    normal: vec3f,
    surfaceVelocity: vec3f,
}

@group(1) @binding(11) var<storage, read> kinematicColliders:
    array<KinematicCollider>;
@group(1) @binding(12) var<storage, read> kinematicColliderFields:
    array<u32>;

fn invalidKinematicColliderFieldSample() -> KinematicColliderFieldSample {
    return KinematicColliderFieldSample(
        KINEMATIC_COLLIDER_NO_CONTACT_DISTANCE,
        vec3f(0.0),
        false,
    );
}

fn readKinematicColliderFieldVoxel(
    colliderIndex: u32,
    voxelIndex: u32,
) -> f32 {
    let voxelsPerCollider = KINEMATIC_COLLIDER_SDF_RESOLUTION
        * KINEMATIC_COLLIDER_SDF_RESOLUTION
        * KINEMATIC_COLLIDER_SDF_RESOLUTION;
    let globalVoxelIndex =
        colliderIndex * voxelsPerCollider + voxelIndex;
    let packed = unpack2x16float(
        kinematicColliderFields[globalVoxelIndex >> 1u],
    );

    return select(
        packed.x,
        packed.y,
        (globalVoxelIndex & 1u) != 0u,
    );
}

fn sampleKinematicColliderField(
    colliderIndex: u32,
    unitPosition: vec3f,
) -> KinematicColliderFieldSample {
    if (
        any(unitPosition < vec3f(-1.0))
        || any(unitPosition > vec3f(1.0))
    ) {
        return invalidKinematicColliderFieldSample();
    }

    let lastGridIndex =
        f32(KINEMATIC_COLLIDER_SDF_RESOLUTION - 1u);
    let gridPosition =
        (unitPosition * 0.5 + vec3f(0.5)) * lastGridIndex;
    let maximumBaseIndex =
        KINEMATIC_COLLIDER_SDF_RESOLUTION - 2u;
    let baseIndex = min(
        vec3u(floor(gridPosition)),
        vec3u(maximumBaseIndex),
    );
    let interpolation = gridPosition - vec3f(baseIndex);
    let rowStride = KINEMATIC_COLLIDER_SDF_RESOLUTION;
    let layerStride = rowStride * rowStride;

    var values: array<f32, 8>;
    var valueIndex = 0u;
    for (var z = 0u; z <= 1u; z = z + 1u) {
        for (var y = 0u; y <= 1u; y = y + 1u) {
            for (var x = 0u; x <= 1u; x = x + 1u) {
                let gridIndex = baseIndex + vec3u(x, y, z);
                let voxelIndex = gridIndex.x
                    + gridIndex.y * rowStride
                    + gridIndex.z * layerStride;
                values[valueIndex] =
                    readKinematicColliderFieldVoxel(
                        colliderIndex,
                        voxelIndex,
                    );
                valueIndex = valueIndex + 1u;
            }
        }
    }

    let x00 = mix(values[0], values[1], interpolation.x);
    let x10 = mix(values[2], values[3], interpolation.x);
    let x01 = mix(values[4], values[5], interpolation.x);
    let x11 = mix(values[6], values[7], interpolation.x);
    let xy0 = mix(x00, x10, interpolation.y);
    let xy1 = mix(x01, x11, interpolation.y);
    let signedDistance = mix(xy0, xy1, interpolation.z);

    let distanceDx0 = mix(
        values[1] - values[0],
        values[3] - values[2],
        interpolation.y,
    );
    let distanceDx1 = mix(
        values[5] - values[4],
        values[7] - values[6],
        interpolation.y,
    );
    let distanceDy0 = mix(
        values[2] - values[0],
        values[3] - values[1],
        interpolation.x,
    );
    let distanceDy1 = mix(
        values[6] - values[4],
        values[7] - values[5],
        interpolation.x,
    );
    let distanceDz0 = mix(
        values[4] - values[0],
        values[5] - values[1],
        interpolation.x,
    );
    let distanceDz1 = mix(
        values[6] - values[2],
        values[7] - values[3],
        interpolation.x,
    );
    let gridToUnitScale = lastGridIndex * 0.5;
    let gradient = vec3f(
        mix(distanceDx0, distanceDx1, interpolation.z),
        mix(distanceDy0, distanceDy1, interpolation.z),
        mix(distanceDz0, distanceDz1, interpolation.y),
    ) * gridToUnitScale;

    return KinematicColliderFieldSample(
        signedDistance,
        gradient,
        true,
    );
}

fn sampleKinematicCollider(
    collider: KinematicCollider,
    colliderIndex: u32,
    worldPosition: vec3f,
) -> KinematicColliderSample {
    let unitPosition = (
        collider.currentUnitFromWorld
        * vec4f(worldPosition, 1.0)
    ).xyz;
    let fieldSample = sampleKinematicColliderField(
        colliderIndex,
        unitPosition,
    );
    if !fieldSample.valid {
        return KinematicColliderSample(
            KINEMATIC_COLLIDER_NO_CONTACT_DISTANCE,
            vec3f(0.0),
            vec3f(0.0),
        );
    }

    let worldGradient = (
        transpose(collider.currentUnitFromWorld)
        * vec4f(fieldSample.gradient, 0.0)
    ).xyz;
    let worldGradientLengthSquared =
        dot(worldGradient, worldGradient);
    if worldGradientLengthSquared <= KINEMATIC_COLLIDER_EPSILON {
        return KinematicColliderSample(
            KINEMATIC_COLLIDER_NO_CONTACT_DISTANCE,
            vec3f(0.0),
            vec3f(0.0),
        );
    }

    let normal = worldGradient * inverseSqrt(
        worldGradientLengthSquared,
    );
    let currentSurfaceWorld =
        worldPosition - fieldSample.signedDistance * normal;
    let surfaceUnitPosition = (
        collider.currentUnitFromWorld
        * vec4f(currentSurfaceWorld, 1.0)
    );
    let transformedCurrentSurface = (
        collider.currentWorldFromUnit * surfaceUnitPosition
    ).xyz;
    let transformedPreviousSurface = (
        collider.previousWorldFromUnit * surfaceUnitPosition
    ).xyz;
    let inverseTimestep = 1.0 / max(
        collider.timing.x,
        KINEMATIC_COLLIDER_EPSILON,
    );

    return KinematicColliderSample(
        fieldSample.signedDistance,
        normal,
        (
            transformedCurrentSurface
            - transformedPreviousSurface
        ) * inverseTimestep,
    );
}

fn sampleKinematicColliderUnion(
    worldPosition: vec3f,
) -> KinematicColliderSample {
    var closestSample = KinematicColliderSample(
        KINEMATIC_COLLIDER_NO_CONTACT_DISTANCE,
        vec3f(0.0),
        vec3f(0.0),
    );

    for (
        var colliderIndex = 0u;
        colliderIndex < KINEMATIC_COLLIDER_COUNT;
        colliderIndex = colliderIndex + 1u
    ) {
        let sample = sampleKinematicCollider(
            kinematicColliders[colliderIndex],
            colliderIndex,
            worldPosition,
        );
        if sample.signedDistance < closestSample.signedDistance {
            closestSample = sample;
        }
    }

    return closestSample;
}

fn projectVelocityAgainstKinematicColliderSample(
    sample: KinematicColliderSample,
    velocity: vec3f,
) -> vec3f {
    if sample.signedDistance > 0.0 {
        return velocity;
    }

    let relativeVelocity = velocity - sample.surfaceVelocity;
    let normalVelocity = dot(relativeVelocity, sample.normal);
    if normalVelocity >= 0.0 {
        return velocity;
    }

    let tangentialVelocity =
        relativeVelocity - normalVelocity * sample.normal;
    let tangentialSpeed = length(tangentialVelocity);
    let frictionSpeed =
        max(uniforms.colliderFriction, 0.0) * -normalVelocity;
    let tangentialScale = max(
        0.0,
        1.0 - frictionSpeed / max(
            tangentialSpeed,
            KINEMATIC_COLLIDER_EPSILON,
        ),
    );

    return sample.surfaceVelocity
        + tangentialScale * tangentialVelocity;
}

fn projectVelocityAgainstKinematicCollider(
    worldPosition: vec3f,
    velocity: vec3f,
) -> vec3f {
    return projectVelocityAgainstKinematicColliderSample(
        sampleKinematicColliderUnion(worldPosition),
        velocity,
    );
}

fn resolveParticlePositionAgainstKinematicCollider(
    worldPosition: vec3f,
) -> vec3f {
    let sample = sampleKinematicColliderUnion(worldPosition);
    if sample.signedDistance > 0.0 {
        return worldPosition;
    }

    return worldPosition - sample.signedDistance * sample.normal;
}
