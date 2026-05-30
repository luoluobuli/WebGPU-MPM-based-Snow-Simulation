const COLLIDER_SDF_RESOLUTION = 64u;
const COLLIDER_SDF_LAST_INDEX = 63u;
const COLLIDER_SDF_LAST_COORD = 63.0;
const COLLIDER_SDF_MAX_SWEEP_STEPS = 16u;

@group(1) @binding(10) var<storage, read> colliderSdfData: array<f32>;

fn colliderSdfCellSize() -> vec3f {
    return (uniforms.colliderMaxCoords - uniforms.colliderMinCoords) / vec3f(COLLIDER_SDF_LAST_COORD);
}

fn colliderSdfIndex(cell: vec3u) -> u32 {
    return cell.x
        + cell.y * COLLIDER_SDF_RESOLUTION
        + cell.z * COLLIDER_SDF_RESOLUTION * COLLIDER_SDF_RESOLUTION;
}

fn sampleColliderSdfGrid(grid_pos_in: vec3f) -> f32 {
    let grid_pos = clamp(grid_pos_in, vec3f(0.0), vec3f(COLLIDER_SDF_LAST_COORD));
    let base_f = floor(grid_pos);
    let base = vec3u(u32(base_f.x), u32(base_f.y), u32(base_f.z));
    let next = min(base + vec3u(1u), vec3u(COLLIDER_SDF_LAST_INDEX));
    let frac = grid_pos - vec3f(f32(base.x), f32(base.y), f32(base.z));

    let c000 = colliderSdfData[colliderSdfIndex(vec3u(base.x, base.y, base.z))];
    let c100 = colliderSdfData[colliderSdfIndex(vec3u(next.x, base.y, base.z))];
    let c010 = colliderSdfData[colliderSdfIndex(vec3u(base.x, next.y, base.z))];
    let c110 = colliderSdfData[colliderSdfIndex(vec3u(next.x, next.y, base.z))];
    let c001 = colliderSdfData[colliderSdfIndex(vec3u(base.x, base.y, next.z))];
    let c101 = colliderSdfData[colliderSdfIndex(vec3u(next.x, base.y, next.z))];
    let c011 = colliderSdfData[colliderSdfIndex(vec3u(base.x, next.y, next.z))];
    let c111 = colliderSdfData[colliderSdfIndex(vec3u(next.x, next.y, next.z))];

    let c00 = mix(c000, c100, frac.x);
    let c10 = mix(c010, c110, frac.x);
    let c01 = mix(c001, c101, frac.x);
    let c11 = mix(c011, c111, frac.x);
    let c0 = mix(c00, c10, frac.y);
    let c1 = mix(c01, c11, frac.y);
    return mix(c0, c1, frac.z);
}

fn sampleColliderSdfNearest(local_pos: vec3f) -> f32 {
    let sdf_min = uniforms.colliderMinCoords;
    let sdf_max = uniforms.colliderMaxCoords;
    let sdf_extent = sdf_max - sdf_min;
    if any(sdf_extent <= vec3f(1e-6)) {
        return 1e6;
    }

    let grid_pos = (local_pos - sdf_min) / sdf_extent * vec3f(COLLIDER_SDF_LAST_COORD);
    let clamped_grid_pos = clamp(floor(grid_pos + vec3f(0.5)), vec3f(0.0), vec3f(COLLIDER_SDF_LAST_COORD));
    let clamped_cell = vec3u(
        u32(clamped_grid_pos.x),
        u32(clamped_grid_pos.y),
        u32(clamped_grid_pos.z),
    );
    let clamped_local_pos = sdf_min + clamped_grid_pos / vec3f(COLLIDER_SDF_LAST_COORD) * sdf_extent;
    return colliderSdfData[colliderSdfIndex(clamped_cell)] + length(local_pos - clamped_local_pos);
}

fn sampleColliderSdf(local_pos: vec3f) -> f32 {
    let sdf_min = uniforms.colliderMinCoords;
    let sdf_max = uniforms.colliderMaxCoords;
    let sdf_extent = sdf_max - sdf_min;
    if any(sdf_extent <= vec3f(1e-6)) {
        return 1e6;
    }

    let grid_pos = (local_pos - sdf_min) / sdf_extent * vec3f(COLLIDER_SDF_LAST_COORD);
    let clamped_grid_pos = clamp(grid_pos, vec3f(0.0), vec3f(COLLIDER_SDF_LAST_COORD));
    let clamped_local_pos = sdf_min + clamped_grid_pos / vec3f(COLLIDER_SDF_LAST_COORD) * sdf_extent;

    return sampleColliderSdfGrid(clamped_grid_pos) + length(local_pos - clamped_local_pos);
}

fn colliderSdfGradient(local_pos: vec3f) -> vec3f {
    let cell_size = colliderSdfCellSize();
    let dx = sampleColliderSdf(local_pos + vec3f(cell_size.x, 0.0, 0.0))
        - sampleColliderSdf(local_pos - vec3f(cell_size.x, 0.0, 0.0));
    let dy = sampleColliderSdf(local_pos + vec3f(0.0, cell_size.y, 0.0))
        - sampleColliderSdf(local_pos - vec3f(0.0, cell_size.y, 0.0));
    let dz = sampleColliderSdf(local_pos + vec3f(0.0, 0.0, cell_size.z))
        - sampleColliderSdf(local_pos - vec3f(0.0, 0.0, cell_size.z));

    let grad = vec3f(dx / max(cell_size.x, 1e-6), dy / max(cell_size.y, 1e-6), dz / max(cell_size.z, 1e-6));
    let grad_len = length(grad);
    if grad_len < 1e-6 {
        return vec3f(0.0, 0.0, 1.0);
    }

    return grad / grad_len;
}

fn segmentOutsideColliderSdfBounds(prev_local: vec3f, current_local: vec3f, margin: f32) -> bool {
    let path_min = min(prev_local, current_local);
    let path_max = max(prev_local, current_local);
    let bounds_min = uniforms.colliderMinCoords - vec3f(margin);
    let bounds_max = uniforms.colliderMaxCoords + vec3f(margin);

    return any(path_min > bounds_max) || any(path_max < bounds_min);
}

fn resolveParticleCollision(particle: ptr<function, ParticleData>) {
    let transform = uniforms.colliderTransformMat;
    let inv_transform = uniforms.colliderTransformInv;

    let current_pos = (*particle).pos;
    let prev_pos = current_pos - (*particle).pos_displacement;
    let current_local = (inv_transform * vec4f(current_pos, 1.0)).xyz;
    let prev_local = (inv_transform * vec4f(prev_pos, 1.0)).xyz;
    let local_displacement = current_local - prev_local;

    let cell_size = colliderSdfCellSize();
    let max_cell_size = max(max(cell_size.x, cell_size.y), cell_size.z);
    let bounds_margin = max_cell_size * 2.0;
    if segmentOutsideColliderSdfBounds(prev_local, current_local, bounds_margin) {
        return;
    }

    let travel_len = length(local_displacement);
    let current_phi_fast = sampleColliderSdfNearest(current_local);
    if current_phi_fast > travel_len + max_cell_size {
        return;
    }

    let current_phi = sampleColliderSdf(current_local);
    var hit_local = current_local;
    var hit_phi = current_phi;
    var has_hit = current_phi < 0.0;

    if !has_hit {
        let prev_phi = sampleColliderSdfNearest(prev_local);
        if prev_phi < 0.0 {
            return;
        }

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
                hit_phi = sampleColliderSdf(candidate_local);
                has_hit = true;
                break;
            }
        }
    }

    if !has_hit {
        return;
    }

    let normal_local = colliderSdfGradient(hit_local);
    var normal_world = (transform * vec4f(normal_local, 0.0)).xyz;
    let normal_world_len = length(normal_world);
    if normal_world_len < 1e-6 {
        return;
    }
    normal_world = normal_world / normal_world_len;

    let corrected_local = hit_local - hit_phi * normal_local;
    (*particle).pos = (transform * vec4f(corrected_local, 1.0)).xyz;

    let velocity_scale_fac = 0.2 / uniforms.simulationTimestep;
    let old_vel = (*particle).pos_displacement / uniforms.simulationTimestep;
    let v_rel = old_vel - uniforms.colliderVelocity;
    let vn = dot(v_rel, normal_world);
    let v_n = vn * normal_world;
    let v_t = v_rel - v_n;
    let friction = uniforms.colliderFriction;

    var new_vel: vec3f;
    if vn < 0.0 {
        new_vel = v_t * (1.0 - friction) + uniforms.colliderVelocity * velocity_scale_fac;
    } else {
        new_vel = v_rel + uniforms.colliderVelocity * velocity_scale_fac;
    }

    (*particle).vel = new_vel;
    (*particle).pos_displacement = new_vel * uniforms.simulationTimestep;
}
