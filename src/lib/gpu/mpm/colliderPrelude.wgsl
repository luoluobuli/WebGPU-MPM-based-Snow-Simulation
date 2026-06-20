override COLLIDER_SDF_RESOLUTION: u32 = 64u;
const COLLIDER_SDF_MAX_SWEEP_STEPS = 16u;

fn colliderSdfLastIndex() -> u32 {
    return COLLIDER_SDF_RESOLUTION - 1u;
}

fn colliderSdfLastCoord() -> f32 {
    return f32(colliderSdfLastIndex());
}

override COLLIDER_TRANSFORM_ALWAYS_IDENTITY: u32 = 0u;
override COLLIDER_VELOCITY_ALWAYS_ZERO: u32 = 0u;
override COLLIDER_SDF_ALWAYS_VALID: u32 = 0u;

@group(1) @binding(10) var<storage, read> colliderSdfData: array<f32>;

struct ColliderSdfValueGradient {
    value: f32,
    gradient: vec3f,
    has_gradient: u32,
}

struct ColliderSdfGridValueGradient {
    value: f32,
    gradient_grid: vec3f,
}

fn colliderSdfCellSize() -> vec3f {
    return uniforms.colliderSdfCellSize;
}

fn colliderSdfIsValid() -> bool {
    if COLLIDER_SDF_ALWAYS_VALID != 0u {
        return true;
    }

    return uniforms.colliderSdfValid != 0u;
}

fn colliderTransformIsIdentity() -> bool {
    if COLLIDER_TRANSFORM_ALWAYS_IDENTITY != 0u {
        return true;
    }

    return uniforms.colliderTransformIsIdentity != 0u;
}

fn colliderVelocityIsZero() -> bool {
    if COLLIDER_VELOCITY_ALWAYS_ZERO != 0u {
        return true;
    }

    return uniforms.colliderVelocityIsZero != 0u;
}

fn colliderSdfIndex(cell: vec3u) -> u32 {
    return cell.x
        + cell.y * COLLIDER_SDF_RESOLUTION
        + cell.z * COLLIDER_SDF_RESOLUTION * COLLIDER_SDF_RESOLUTION;
}

fn sampleColliderSdfGridClamped(grid_pos: vec3f) -> f32 {
    let base_f = floor(grid_pos);
    let base = vec3u(u32(base_f.x), u32(base_f.y), u32(base_f.z));
    let next = min(base + vec3u(1u), vec3u(colliderSdfLastIndex()));
    let frac = grid_pos - vec3f(f32(base.x), f32(base.y), f32(base.z));
    let base_y_offset = base.y * COLLIDER_SDF_RESOLUTION;
    let next_y_offset = next.y * COLLIDER_SDF_RESOLUTION;
    let base_z_offset = base.z * COLLIDER_SDF_RESOLUTION * COLLIDER_SDF_RESOLUTION;
    let next_z_offset = next.z * COLLIDER_SDF_RESOLUTION * COLLIDER_SDF_RESOLUTION;
    let base_yz_offset = base_y_offset + base_z_offset;
    let next_y_base_z_offset = next_y_offset + base_z_offset;
    let base_y_next_z_offset = base_y_offset + next_z_offset;
    let next_yz_offset = next_y_offset + next_z_offset;

    let c000 = colliderSdfData[base.x + base_yz_offset];
    let c100 = colliderSdfData[next.x + base_yz_offset];
    let c010 = colliderSdfData[base.x + next_y_base_z_offset];
    let c110 = colliderSdfData[next.x + next_y_base_z_offset];
    let c001 = colliderSdfData[base.x + base_y_next_z_offset];
    let c101 = colliderSdfData[next.x + base_y_next_z_offset];
    let c011 = colliderSdfData[base.x + next_yz_offset];
    let c111 = colliderSdfData[next.x + next_yz_offset];

    let c00 = mix(c000, c100, frac.x);
    let c10 = mix(c010, c110, frac.x);
    let c01 = mix(c001, c101, frac.x);
    let c11 = mix(c011, c111, frac.x);
    let c0 = mix(c00, c10, frac.y);
    let c1 = mix(c01, c11, frac.y);
    return mix(c0, c1, frac.z);
}

fn sampleColliderSdfGridValueGradientClamped(grid_pos: vec3f) -> ColliderSdfGridValueGradient {
    let base_f = floor(grid_pos);
    let base = vec3u(u32(base_f.x), u32(base_f.y), u32(base_f.z));
    let next = min(base + vec3u(1u), vec3u(colliderSdfLastIndex()));
    let frac = grid_pos - vec3f(f32(base.x), f32(base.y), f32(base.z));
    let base_y_offset = base.y * COLLIDER_SDF_RESOLUTION;
    let next_y_offset = next.y * COLLIDER_SDF_RESOLUTION;
    let base_z_offset = base.z * COLLIDER_SDF_RESOLUTION * COLLIDER_SDF_RESOLUTION;
    let next_z_offset = next.z * COLLIDER_SDF_RESOLUTION * COLLIDER_SDF_RESOLUTION;
    let base_yz_offset = base_y_offset + base_z_offset;
    let next_y_base_z_offset = next_y_offset + base_z_offset;
    let base_y_next_z_offset = base_y_offset + next_z_offset;
    let next_yz_offset = next_y_offset + next_z_offset;

    let c000 = colliderSdfData[base.x + base_yz_offset];
    let c100 = colliderSdfData[next.x + base_yz_offset];
    let c010 = colliderSdfData[base.x + next_y_base_z_offset];
    let c110 = colliderSdfData[next.x + next_y_base_z_offset];
    let c001 = colliderSdfData[base.x + base_y_next_z_offset];
    let c101 = colliderSdfData[next.x + base_y_next_z_offset];
    let c011 = colliderSdfData[base.x + next_yz_offset];
    let c111 = colliderSdfData[next.x + next_yz_offset];

    let c00 = mix(c000, c100, frac.x);
    let c10 = mix(c010, c110, frac.x);
    let c01 = mix(c001, c101, frac.x);
    let c11 = mix(c011, c111, frac.x);
    let c0 = mix(c00, c10, frac.y);
    let c1 = mix(c01, c11, frac.y);
    let value = mix(c0, c1, frac.z);

    let dx = mix(
        mix(c100 - c000, c110 - c010, frac.y),
        mix(c101 - c001, c111 - c011, frac.y),
        frac.z
    );
    let dy = mix(
        mix(c010 - c000, c110 - c100, frac.x),
        mix(c011 - c001, c111 - c101, frac.x),
        frac.z
    );
    let dz = mix(
        mix(c001 - c000, c101 - c100, frac.x),
        mix(c011 - c010, c111 - c110, frac.x),
        frac.y
    );

    return ColliderSdfGridValueGradient(value, vec3f(dx, dy, dz));
}

fn sampleColliderSdfGrid(grid_pos_in: vec3f) -> f32 {
    return sampleColliderSdfGridClamped(clamp(grid_pos_in, vec3f(0.0), vec3f(colliderSdfLastCoord())));
}

fn sampleColliderSdfNearest(local_pos: vec3f) -> f32 {
    if !colliderSdfIsValid() {
        return 1e6;
    }

    let sdf_min = uniforms.colliderMinCoords;
    let grid_pos = (local_pos - sdf_min) * uniforms.colliderSdfGridScale;
    let clamped_grid_pos = clamp(floor(grid_pos + vec3f(0.5)), vec3f(0.0), vec3f(colliderSdfLastCoord()));
    let clamped_cell = vec3u(
        u32(clamped_grid_pos.x),
        u32(clamped_grid_pos.y),
        u32(clamped_grid_pos.z),
    );
    let clamped_local_pos = sdf_min + clamped_grid_pos * uniforms.colliderSdfCellSize;
    return colliderSdfData[colliderSdfIndex(clamped_cell)] + length(local_pos - clamped_local_pos);
}

fn sampleColliderSdf(local_pos: vec3f) -> f32 {
    if !colliderSdfIsValid() {
        return 1e6;
    }

    let sdf_min = uniforms.colliderMinCoords;
    let grid_pos = (local_pos - sdf_min) * uniforms.colliderSdfGridScale;
    if all(grid_pos >= vec3f(0.0)) && all(grid_pos <= vec3f(colliderSdfLastCoord())) {
        return sampleColliderSdfGridClamped(grid_pos);
    }

    let clamped_grid_pos = clamp(grid_pos, vec3f(0.0), vec3f(colliderSdfLastCoord()));
    let clamped_local_pos = sdf_min + clamped_grid_pos * uniforms.colliderSdfCellSize;

    return sampleColliderSdfGridClamped(clamped_grid_pos) + length(local_pos - clamped_local_pos);
}

fn sampleColliderSdfValueGradient(local_pos: vec3f) -> ColliderSdfValueGradient {
    let grid_pos = (local_pos - uniforms.colliderMinCoords) * uniforms.colliderSdfGridScale;
    if all(grid_pos > vec3f(0.0)) && all(grid_pos < vec3f(colliderSdfLastCoord())) {
        let sample = sampleColliderSdfGridValueGradientClamped(grid_pos);
        let grad = sample.gradient_grid * uniforms.colliderSdfGridScale;
        let grad_len_squared = dot(grad, grad);
        if grad_len_squared >= 1e-12 {
            return ColliderSdfValueGradient(
                sample.value,
                grad * inverseSqrt(grad_len_squared),
                1u
            );
        }

        return ColliderSdfValueGradient(sample.value, vec3f(0.0), 0u);
    }

    return ColliderSdfValueGradient(sampleColliderSdf(local_pos), vec3f(0.0), 0u);
}

fn sampleColliderSdfCurrentPosition(local_pos: vec3f) -> ColliderSdfValueGradient {
    let grid_pos = (local_pos - uniforms.colliderMinCoords) * uniforms.colliderSdfGridScale;
    if all(grid_pos > vec3f(0.0)) && all(grid_pos < vec3f(colliderSdfLastCoord())) {
        let base_f = floor(grid_pos);
        let base = vec3u(u32(base_f.x), u32(base_f.y), u32(base_f.z));
        let next = min(base + vec3u(1u), vec3u(colliderSdfLastIndex()));
        let frac = grid_pos - vec3f(f32(base.x), f32(base.y), f32(base.z));
        let base_y_offset = base.y * COLLIDER_SDF_RESOLUTION;
        let next_y_offset = next.y * COLLIDER_SDF_RESOLUTION;
        let base_z_offset = base.z * COLLIDER_SDF_RESOLUTION * COLLIDER_SDF_RESOLUTION;
        let next_z_offset = next.z * COLLIDER_SDF_RESOLUTION * COLLIDER_SDF_RESOLUTION;
        let base_yz_offset = base_y_offset + base_z_offset;
        let next_y_base_z_offset = next_y_offset + base_z_offset;
        let base_y_next_z_offset = base_y_offset + next_z_offset;
        let next_yz_offset = next_y_offset + next_z_offset;

        let c000 = colliderSdfData[base.x + base_yz_offset];
        let c100 = colliderSdfData[next.x + base_yz_offset];
        let c010 = colliderSdfData[base.x + next_y_base_z_offset];
        let c110 = colliderSdfData[next.x + next_y_base_z_offset];
        let c001 = colliderSdfData[base.x + base_y_next_z_offset];
        let c101 = colliderSdfData[next.x + base_y_next_z_offset];
        let c011 = colliderSdfData[base.x + next_yz_offset];
        let c111 = colliderSdfData[next.x + next_yz_offset];

        let c00 = mix(c000, c100, frac.x);
        let c10 = mix(c010, c110, frac.x);
        let c01 = mix(c001, c101, frac.x);
        let c11 = mix(c011, c111, frac.x);
        let c0 = mix(c00, c10, frac.y);
        let c1 = mix(c01, c11, frac.y);
        let value = mix(c0, c1, frac.z);

        if value >= 0.0 {
            return ColliderSdfValueGradient(value, vec3f(0.0), 0u);
        }

        let dx = mix(
            mix(c100 - c000, c110 - c010, frac.y),
            mix(c101 - c001, c111 - c011, frac.y),
            frac.z
        );
        let dy = mix(
            mix(c010 - c000, c110 - c100, frac.x),
            mix(c011 - c001, c111 - c101, frac.x),
            frac.z
        );
        let dz = mix(
            mix(c001 - c000, c101 - c100, frac.x),
            mix(c011 - c010, c111 - c110, frac.x),
            frac.y
        );

        let grad = vec3f(dx, dy, dz) * uniforms.colliderSdfGridScale;
        let grad_len_squared = dot(grad, grad);
        if grad_len_squared >= 1e-12 {
            return ColliderSdfValueGradient(value, grad * inverseSqrt(grad_len_squared), 1u);
        }

        return ColliderSdfValueGradient(value, vec3f(0.0), 0u);
    }

    return ColliderSdfValueGradient(sampleColliderSdf(local_pos), vec3f(0.0), 0u);
}

fn colliderSdfGradient(local_pos: vec3f) -> vec3f {
    let grid_pos = (local_pos - uniforms.colliderMinCoords) * uniforms.colliderSdfGridScale;
    if all(grid_pos > vec3f(0.0)) && all(grid_pos < vec3f(colliderSdfLastCoord())) {
        let grad = sampleColliderSdfGridValueGradientClamped(grid_pos).gradient_grid * uniforms.colliderSdfGridScale;
        let grad_len_squared = dot(grad, grad);
        if grad_len_squared >= 1e-12 {
            return grad * inverseSqrt(grad_len_squared);
        }
    }

    let cell_size = colliderSdfCellSize();
    let dx = sampleColliderSdf(local_pos + vec3f(cell_size.x, 0.0, 0.0))
        - sampleColliderSdf(local_pos - vec3f(cell_size.x, 0.0, 0.0));
    let dy = sampleColliderSdf(local_pos + vec3f(0.0, cell_size.y, 0.0))
        - sampleColliderSdf(local_pos - vec3f(0.0, cell_size.y, 0.0));
    let dz = sampleColliderSdf(local_pos + vec3f(0.0, 0.0, cell_size.z))
        - sampleColliderSdf(local_pos - vec3f(0.0, 0.0, cell_size.z));

    let grad = vec3f(dx / max(cell_size.x, 1e-6), dy / max(cell_size.y, 1e-6), dz / max(cell_size.z, 1e-6));
    let grad_len_squared = dot(grad, grad);
    if grad_len_squared < 1e-12 {
        return vec3f(0.0, 0.0, 1.0);
    }

    return grad * inverseSqrt(grad_len_squared);
}

fn segmentOutsideColliderSdfBounds(prev_local: vec3f, current_local: vec3f, margin: f32) -> bool {
    let path_min = min(prev_local, current_local);
    let path_max = max(prev_local, current_local);
    let bounds_min = uniforms.colliderMinCoords - vec3f(margin);
    let bounds_max = uniforms.colliderMaxCoords + vec3f(margin);

    return any(path_min > bounds_max) || any(path_max < bounds_min);
}

fn segmentOutsideColliderWorldBounds(prev_world: vec3f, current_world: vec3f) -> bool {
    let path_min = min(prev_world, current_world);
    let path_max = max(prev_world, current_world);

    return any(path_min > uniforms.colliderWorldMaxCoords) || any(path_max < uniforms.colliderWorldMinCoords);
}

fn resolveParticleCollision(particle: ptr<function, ParticleData>) {
    if !colliderSdfIsValid() {
        return;
    }

    let current_pos = (*particle).pos;
    let prev_pos = current_pos - (*particle).pos_displacement;
    if segmentOutsideColliderWorldBounds(prev_pos, current_pos) {
        return;
    }

    let collider_transform_is_identity = colliderTransformIsIdentity();
    var current_local = current_pos;
    var prev_local = prev_pos;
    if !collider_transform_is_identity {
        let inv_transform = uniforms.colliderTransformInv;
        current_local = (inv_transform * vec4f(current_pos, 1.0)).xyz;
        prev_local = (inv_transform * vec4f(prev_pos, 1.0)).xyz;
    }
    let local_displacement = current_local - prev_local;

    let max_cell_size = uniforms.colliderSdfMaxCellSize;
    let bounds_margin = max_cell_size * 2.0;
    if !collider_transform_is_identity
        && segmentOutsideColliderSdfBounds(prev_local, current_local, bounds_margin)
    {
        return;
    }

    let travel_len_squared = dot(local_displacement, local_displacement);
    let current_phi_fast = sampleColliderSdfNearest(current_local);
    let fast_clearance = current_phi_fast - max_cell_size;
    if fast_clearance > 0.0 && fast_clearance * fast_clearance > travel_len_squared {
        return;
    }

    let current_sample = sampleColliderSdfCurrentPosition(current_local);
    var hit_local = current_local;
    var hit_phi = current_sample.value;
    var hit_normal_local = current_sample.gradient;
    var hit_has_normal = current_sample.has_gradient != 0u;
    var has_hit = current_sample.value < 0.0;

    if !has_hit {
        if travel_len_squared == 0.0 {
            return;
        }

        let prev_phi = sampleColliderSdfNearest(prev_local);
        if prev_phi < 0.0 {
            return;
        }

        let travel_len = sqrt(travel_len_squared);
        let target_step_len = max(max_cell_size * 0.5, 1e-4);
        let sweep_steps = min(
            COLLIDER_SDF_MAX_SWEEP_STEPS,
            max(1u, u32(ceil(travel_len / target_step_len))),
        );

        for (var i = 1u; i < COLLIDER_SDF_MAX_SWEEP_STEPS; i = i + 1u) {
            if i >= sweep_steps {
                break;
            }

            let t = f32(i) / f32(sweep_steps);
            let candidate_local = mix(prev_local, current_local, t);
            let candidate_phi = sampleColliderSdfNearest(candidate_local);
            if candidate_phi < 0.0 {
                hit_local = candidate_local;
                let candidate_sample = sampleColliderSdfValueGradient(candidate_local);
                hit_phi = candidate_sample.value;
                hit_normal_local = candidate_sample.gradient;
                hit_has_normal = candidate_sample.has_gradient != 0u;
                has_hit = true;
                break;
            }
        }
    }

    if !has_hit {
        return;
    }

    var normal_local = hit_normal_local;
    if !hit_has_normal {
        normal_local = colliderSdfGradient(hit_local);
    }
    var normal_world = normal_local;
    if !collider_transform_is_identity {
        normal_world = (uniforms.colliderTransformMat * vec4f(normal_local, 0.0)).xyz;
    }
    let normal_world_len_squared = dot(normal_world, normal_world);
    if normal_world_len_squared < 1e-12 {
        return;
    }
    normal_world = normal_world * inverseSqrt(normal_world_len_squared);

    let corrected_local = hit_local - hit_phi * normal_local;
    if collider_transform_is_identity {
        (*particle).pos = corrected_local;
    } else {
        (*particle).pos = (uniforms.colliderTransformMat * vec4f(corrected_local, 1.0)).xyz;
    }

    let velocity_scale_fac = 0.2 * uniforms.invSimulationTimestep;
    let collider_velocity_is_zero = colliderVelocityIsZero();
    var v_rel = (*particle).vel;
    if !collider_velocity_is_zero {
        v_rel = v_rel - uniforms.colliderVelocity;
    }
    let vn = dot(v_rel, normal_world);
    let v_n = vn * normal_world;
    let v_t = v_rel - v_n;
    let friction = uniforms.colliderFriction;

    var new_vel: vec3f;
    if vn < 0.0 {
        new_vel = v_t * (1.0 - friction);
    } else {
        new_vel = v_rel;
    }
    if !collider_velocity_is_zero {
        new_vel = new_vel + uniforms.colliderVelocity * velocity_scale_fac;
    }

    (*particle).vel = new_vel;
    (*particle).pos_displacement = new_vel * uniforms.simulationTimestep;
}
