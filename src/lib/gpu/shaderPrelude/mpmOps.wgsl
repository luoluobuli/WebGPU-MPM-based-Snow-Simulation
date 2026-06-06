struct ParticleData {
    // 0
    pos: vec3f, // 12
    _hom: f32, // 16; vertex shader expects a vec4
    vel: vec3f, // 28
    mass: f32, // 32
    deformationElastic: mat3x3f, // 80
    deformationPlastic: mat3x3f, // 128
    pos_displacement: vec3f, // 140
    // 144
    deformation_displacement: mat3x3f, // 192
}

const FIXED_POINT_SCALE = 65536.0;
const INV_FIXED_POINT_SCALE = 1.0 / FIXED_POINT_SCALE;
const MAX_FIXED_POINT_I32 = 2147483000.0;
const MAX_FIXED_POINT_GRID_SPEED = MAX_FIXED_POINT_I32 * INV_FIXED_POINT_SCALE;
const DEFAULT_PARTICLE_MASS = 1.0 / 3.0;
const PARTICLE_FLAG_DEFORMATION_DELTA_VALID = 1u;
const PARTICLE_FLAG_ELASTIC_NON_IDENTITY = 2u;
const PARTICLE_FLAG_ANCHORED = 4u;
const PARTICLE_MATERIAL_SHIFT = 8u;
const PARTICLE_MATERIAL_MASK = 3u << PARTICLE_MATERIAL_SHIFT;
const PARTICLE_MATERIAL_DEFAULT = 0u;
const PARTICLE_MATERIAL_SOIL = 1u;
const PARTICLE_MATERIAL_BARK = 2u;
const PARTICLE_MATERIAL_LEAF = 3u;
const PARTICLE_MATRIX_ELASTIC_CHANGED = 1u;
const PARTICLE_MATRIX_PLASTIC_CHANGED = 2u;
const MATERIAL_DAMPING_REFERENCE_TIMESTEP_S = 1.0 / 60.0;
const PARTICLE_SPAWN_ANCHORED_OFFSET = 8.0;

struct QuadraticBSplineStencilVectors {
    x: vec3f,
    y: vec3f,
    z: vec3f,
}

fn materialFlagFromId(material: u32) -> u32 {
    return (material & 3u) << PARTICLE_MATERIAL_SHIFT;
}

fn spawnPointMaterialIsAnchored(encodedMaterial: f32) -> bool {
    return encodedMaterial >= PARTICLE_SPAWN_ANCHORED_OFFSET;
}

fn particleMaterialFromSpawnPoint(encodedMaterial: f32) -> u32 {
    let material = select(
        encodedMaterial,
        encodedMaterial - PARTICLE_SPAWN_ANCHORED_OFFSET,
        spawnPointMaterialIsAnchored(encodedMaterial),
    );

    return u32(clamp(round(material), 0.0, 3.0));
}

fn particleFlagsFromSpawnPoint(encodedMaterial: f32) -> u32 {
    let material = particleMaterialFromSpawnPoint(encodedMaterial);
    var flags = materialFlagFromId(material);
    if spawnPointMaterialIsAnchored(encodedMaterial) {
        flags = flags | PARTICLE_FLAG_ANCHORED;
    }

    return flags;
}

fn particleMaterial(flags: u32) -> u32 {
    return (flags & PARTICLE_MATERIAL_MASK) >> PARTICLE_MATERIAL_SHIFT;
}

fn particleIsAnchored(flags: u32) -> bool {
    return (flags & PARTICLE_FLAG_ANCHORED) != 0u;
}

fn particlePersistentFlags(flags: u32) -> u32 {
    return flags & (
        PARTICLE_FLAG_ELASTIC_NON_IDENTITY
        | PARTICLE_FLAG_ANCHORED
        | PARTICLE_MATERIAL_MASK
    );
}

fn particleMassForMaterial(material: u32) -> f32 {
    if material == PARTICLE_MATERIAL_SOIL {
        return 0.82;
    }
    if material == PARTICLE_MATERIAL_BARK {
        return 0.48;
    }
    if material == PARTICLE_MATERIAL_LEAF {
        return 0.18;
    }

    return DEFAULT_PARTICLE_MASS;
}

fn materialVelocityDampingPerReferenceStep(material: u32) -> vec3f {
    if material == PARTICLE_MATERIAL_SOIL {
        return vec3f(0.72, 0.72, 0.82);
    }
    if material == PARTICLE_MATERIAL_BARK {
        return vec3f(0.93, 0.93, 0.96);
    }
    if material == PARTICLE_MATERIAL_LEAF {
        return vec3f(0.86, 0.86, 0.9);
    }

    return vec3f(1.0);
}

fn materialVelocityDamping(material: u32) -> vec3f {
    // Damping values are authored as 60 Hz frame retention. Scale them to the
    // current substep so CFL subdivision does not make gravity look weaker.
    return pow(
        materialVelocityDampingPerReferenceStep(material),
        vec3f(uniforms.simulationTimestep / MATERIAL_DAMPING_REFERENCE_TIMESTEP_S),
    );
}

fn materialBoundaryFriction(material: u32) -> f32 {
    if material == PARTICLE_MATERIAL_SOIL {
        return 0.96;
    }
    if material == PARTICLE_MATERIAL_BARK {
        return 0.91;
    }
    if material == PARTICLE_MATERIAL_LEAF {
        return 0.86;
    }

    return 0.35;
}

fn applyMaterialVelocityDamping(
    particle: ptr<function, ParticleData>,
    material: u32,
) {
    let gravity_delta = uniforms.gravityDeltaVelocity;
    (*particle).vel = ((*particle).vel - gravity_delta)
        * materialVelocityDamping(material)
        + gravity_delta;
}

fn resetAnchoredParticleMotion(particle: ptr<function, ParticleData>) {
    (*particle).vel = vec3f();
    (*particle).pos_displacement = vec3f();
    (*particle).deformation_displacement = mat3x3f();
    (*particle).deformationElastic = IDENTITY_MAT3;
    (*particle).deformationPlastic = IDENTITY_MAT3;
}

fn calculateGridCoordinate(pos: vec3f) -> vec3f {
    return (pos - uniforms.gridMinCoords) * uniforms.invGridCellDims;
}

fn calculateCellNumber(pos: vec3f) -> vec3i {
    return vec3i(floor(calculateGridCoordinate(pos)));
}

fn gridCoordinateCanTouchGrid(grid_coord: vec3f) -> bool {
    return all(grid_coord >= vec3f(0.0))
        && all(grid_coord < vec3f(uniforms.gridResolution));
}

fn isFiniteScalar(value: f32) -> bool {
    return value == value && abs(value) < 1e20;
}

fn isFiniteVec3(value: vec3f) -> bool {
    return isFiniteScalar(value.x) && isFiniteScalar(value.y) && isFiniteScalar(value.z);
}

fn isFiniteMat3(value: mat3x3f) -> bool {
    return isFiniteVec3(value[0]) && isFiniteVec3(value[1]) && isFiniteVec3(value[2]);
}

fn sanitizeScalar(value: f32, fallback: f32) -> f32 {
    if isFiniteScalar(value) {
        return value;
    }

    return fallback;
}

fn sanitizeVec3(value: vec3f, fallback: vec3f) -> vec3f {
    return vec3f(
        sanitizeScalar(value.x, fallback.x),
        sanitizeScalar(value.y, fallback.y),
        sanitizeScalar(value.z, fallback.z),
    );
}

fn maxFixedPointGridSpeed() -> f32 {
    return MAX_FIXED_POINT_GRID_SPEED;
}

fn maxFixedPointGridSpeedSquared() -> f32 {
    return MAX_FIXED_POINT_GRID_SPEED * MAX_FIXED_POINT_GRID_SPEED;
}

fn maxStableParticleDisplacement() -> f32 {
    return uniforms.maxStableParticleDisplacement;
}

fn maxStableParticleDisplacementSquared() -> f32 {
    return uniforms.maxStableParticleDisplacementSquared;
}

fn clampVec3LengthWithMaxSquared(value: vec3f, maxLength: f32, maxLenSquared: f32) -> vec3f {
    let safe_value = sanitizeVec3(value, vec3f(0.0));
    let len_squared = dot(safe_value, safe_value);
    if len_squared > maxLenSquared {
        return safe_value * (maxLength * inverseSqrt(len_squared));
    }

    return safe_value;
}

fn clampVec3Length(value: vec3f, maxLength: f32) -> vec3f {
    return clampVec3LengthWithMaxSquared(value, maxLength, maxLength * maxLength);
}

fn clampVec3LengthNoSanitize(value: vec3f, maxLength: f32) -> vec3f {
    let len_squared = dot(value, value);
    let max_len_squared = maxLength * maxLength;
    if len_squared > max_len_squared {
        return value * (maxLength * inverseSqrt(len_squared));
    }

    return value;
}

fn clampVec3LengthNoSanitizeWithMaxSquared(value: vec3f, maxLength: f32, maxLenSquared: f32) -> vec3f {
    let len_squared = dot(value, value);
    if len_squared > maxLenSquared {
        return value * (maxLength * inverseSqrt(len_squared));
    }

    return value;
}

fn clampMatrixComponents(value: mat3x3f, limit: f32) -> mat3x3f {
    let lower = vec3f(-limit);
    let upper = vec3f(limit);
    return mat3x3f(
        clamp(value[0], lower, upper),
        clamp(value[1], lower, upper),
        clamp(value[2], lower, upper),
    );
}

fn sanitizeNonZeroDeformationDelta(value: mat3x3f) -> mat3x3f {
    if !isFiniteMat3(value) {
        return mat3x3f();
    }

    return clampMatrixComponents(value, 0.35);
}

fn sanitizeDeformationDelta(value: mat3x3f) -> mat3x3f {
    if mat3x3IsZero(value) {
        return value;
    }

    return sanitizeNonZeroDeformationDelta(value);
}

fn sanitizeNonZeroVelocityGradient(value: mat3x3f) -> mat3x3f {
    if !isFiniteMat3(value) {
        return mat3x3f();
    }

    return clampMatrixComponents(value, 0.35 * max(uniforms.invSimulationTimestep, 1.0));
}

fn sanitizeVelocityGradient(value: mat3x3f) -> mat3x3f {
    if mat3x3IsZero(value) {
        return value;
    }

    return sanitizeNonZeroVelocityGradient(value);
}

fn deformationDeltaFromVelocityGradient(velocity_gradient: mat3x3f) -> mat3x3f {
    return sanitizeDeformationDelta(velocity_gradient * uniforms.simulationTimestep);
}

fn simulationDomainMaxInside() -> vec3f {
    return uniforms.simulationDomainMaxInside;
}

fn clampPositionToSimulationDomain(pos: vec3f) -> vec3f {
    return clamp(
        sanitizeVec3(pos, uniforms.simulationDomainCenter),
        uniforms.gridMinCoords,
        uniforms.simulationDomainMaxInside,
    );
}

fn particlePositionCanTouchGrid(pos: vec3f) -> bool {
    return all(uniforms.gridMinCoords <= pos)
        && all(pos < uniforms.gridMaxCoords);
}

fn sanitizeParticleKinematicsWithoutDeformationDelta(particle: ptr<function, ParticleData>) {
    (*particle)._hom = 1.0;
    (*particle).pos = clampPositionToSimulationDomain((*particle).pos);

    if !isFiniteScalar((*particle).mass) || (*particle).mass <= 0.0 {
        (*particle).mass = DEFAULT_PARTICLE_MASS;
    }

    (*particle).vel = clampVec3LengthWithMaxSquared(
        (*particle).vel,
        maxFixedPointGridSpeed(),
        maxFixedPointGridSpeedSquared(),
    );
    (*particle).pos_displacement = clampVec3LengthWithMaxSquared(
        (*particle).pos_displacement,
        maxStableParticleDisplacement(),
        maxStableParticleDisplacementSquared(),
    );
}

fn sanitizeParticleKinematicsWithoutDeformationDeltaWithKnownScalarRepairs(
    particle: ptr<function, ParticleData>,
    write_hom: bool,
    write_mass: bool,
) {
    if write_hom {
        (*particle)._hom = 1.0;
    }
    (*particle).pos = clampPositionToSimulationDomain((*particle).pos);

    if write_mass {
        (*particle).mass = DEFAULT_PARTICLE_MASS;
    }

    (*particle).vel = clampVec3LengthWithMaxSquared(
        (*particle).vel,
        maxFixedPointGridSpeed(),
        maxFixedPointGridSpeedSquared(),
    );
    (*particle).pos_displacement = clampVec3LengthWithMaxSquared(
        (*particle).pos_displacement,
        maxStableParticleDisplacement(),
        maxStableParticleDisplacementSquared(),
    );
}

fn sanitizeParticleKinematics(particle: ptr<function, ParticleData>) {
    sanitizeParticleKinematicsWithoutDeformationDelta(particle);
    (*particle).deformation_displacement = sanitizeVelocityGradient((*particle).deformation_displacement);
}

fn sanitizeParticleMatricesAndGetChangedFlags(particle: ptr<function, ParticleData>) -> u32 {
    var changed_flags = 0u;

    if !mat3x3IsIdentity((*particle).deformationElastic) {
        let elastic_det = determinant((*particle).deformationElastic);
        if !isFiniteMat3((*particle).deformationElastic)
            || elastic_det != elastic_det
            || elastic_det < 0.05
            || elastic_det > 20.0
        {
            (*particle).deformationElastic = IDENTITY_MAT3;
            (*particle).deformation_displacement = mat3x3f();
            changed_flags |= PARTICLE_MATRIX_ELASTIC_CHANGED;
        }
    }

    if !mat3x3IsIdentity((*particle).deformationPlastic) {
        let plastic_det = determinant((*particle).deformationPlastic);
        if !isFiniteMat3((*particle).deformationPlastic)
            || plastic_det != plastic_det
            || plastic_det < 0.05
            || plastic_det > 20.0
        {
            (*particle).deformationPlastic = IDENTITY_MAT3;
            changed_flags |= PARTICLE_MATRIX_PLASTIC_CHANGED;
        }
    }

    return changed_flags;
}

fn sanitizeParticleMatrices(particle: ptr<function, ParticleData>) {
    sanitizeParticleMatricesAndGetChangedFlags(particle);
}

fn sanitizeParticle(particle: ptr<function, ParticleData>) {
    sanitizeParticleKinematics(particle);
    sanitizeParticleMatrices(particle);
}

fn toFixedPointI32(value: f32) -> i32 {
    let scaled_value = sanitizeScalar(value, 0.0) * FIXED_POINT_SCALE;
    return i32(clamp(scaled_value, -2147483000.0, 2147483000.0));
}

fn toFixedPointI32NoSanitize(value: f32) -> i32 {
    return i32(clamp(value * FIXED_POINT_SCALE, -2147483000.0, 2147483000.0));
}

fn cellNumberInGridRange(cellNumber: vec3i) -> bool {
    return all(vec3i(0) <= cellNumber) && all(cellNumber < vec3i(uniforms.gridResolution));
}

fn linearizeCellIndex(cellNumber: vec3u) -> u32 {
    return cellNumber.x + uniforms.gridResolution.x * (cellNumber.y + uniforms.gridResolution.y * cellNumber.z);
}

fn calculateQuadraticBSplineCellWeights(fractionalPosFromCellMin: vec3f) -> array<vec3f, 3> {
    var weights: array<vec3f, 3>;

    let one_minus_pos = vec3f(1.0) - fractionalPosFromCellMin;
    let centered_pos = fractionalPosFromCellMin - vec3f(0.5);
    weights[0] = 0.5 * one_minus_pos * one_minus_pos;
    weights[1] = vec3f(0.75) - centered_pos * centered_pos;
    weights[2] = 0.5 * fractionalPosFromCellMin * fractionalPosFromCellMin;

    return weights;
}

fn calculateQuadraticBSplineCellWeightVectors(fractionalPosFromCellMin: vec3f) -> QuadraticBSplineStencilVectors {
    let one_minus_pos = vec3f(1.0) - fractionalPosFromCellMin;
    let centered_pos = fractionalPosFromCellMin - vec3f(0.5);
    let weight0 = 0.5 * one_minus_pos * one_minus_pos;
    let weight1 = vec3f(0.75) - centered_pos * centered_pos;
    let weight2 = 0.5 * fractionalPosFromCellMin * fractionalPosFromCellMin;

    return QuadraticBSplineStencilVectors(
        vec3f(weight0.x, weight1.x, weight2.x),
        vec3f(weight0.y, weight1.y, weight2.y),
        vec3f(weight0.z, weight1.z, weight2.z),
    );
}

fn calculateQuadraticBSplineCellWeightDerivatives(fractionalPosFromCellMin: vec3f) -> array<vec3f, 3> {
    var derivatives: array<vec3f, 3>;

    // derivative of B-spline weights wrt fractional pos
    derivatives[0] = fractionalPosFromCellMin - vec3f(1.0);
    derivatives[1] = vec3f(1.0) - 2.0 * fractionalPosFromCellMin;
    derivatives[2] = fractionalPosFromCellMin;

    return derivatives;
}

fn calculateQuadraticBSplineCellWeightDerivativeVectors(
    fractionalPosFromCellMin: vec3f,
    inv_grid_cell_dims: vec3f,
) -> QuadraticBSplineStencilVectors {
    let derivative0 = (fractionalPosFromCellMin - vec3f(1.0)) * inv_grid_cell_dims;
    let derivative1 = (vec3f(1.0) - 2.0 * fractionalPosFromCellMin) * inv_grid_cell_dims;
    let derivative2 = fractionalPosFromCellMin * inv_grid_cell_dims;

    return QuadraticBSplineStencilVectors(
        vec3f(derivative0.x, derivative1.x, derivative2.x),
        vec3f(derivative0.y, derivative1.y, derivative2.y),
        vec3f(derivative0.z, derivative1.z, derivative2.z),
    );
}
