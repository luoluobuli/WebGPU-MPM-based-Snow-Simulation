@group(1) @binding(0) var<storage, read_write> mass_grid: array<atomic<u32>>;
@group(1) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(2) var depthTexture: texture_storage_2d<r32float, write>;
@group(1) @binding(3) var environmentTexture: texture_2d<f32>;
@group(1) @binding(4) var environmentSampler: sampler;

const EXTINCTION_COEFFICIENT = 724.;
const SCATTERING_ALBEDO = vec3f(0.9, 0.985, 0.99);
const STEP_SIZE = 0.0125;
const N_MAX_STEPS = 256u;
const SHADOW_STEP_SIZE = STEP_SIZE;
const N_MAX_SHADOW_STEPS = 64u;

fn readDensity(worldPos: vec3f) -> f32 {
    if any(worldPos < uniforms.gridMinCoords) || any(worldPos >= uniforms.gridMaxCoords) {
        return 0;
    }

    let gridRange = uniforms.gridMaxCoords - uniforms.gridMinCoords;
    let gridRes = vec3f(uniforms.gridResolution);
    let cellSize = gridRange / gridRes;
    
    let localPos = worldPos - uniforms.gridMinCoords;
    let gridPos = localPos / cellSize;
    
    let splatPos = gridPos - 0.5;
    let start_cell_number = vec3u(splatPos);
    let fractional_pos = splatPos - vec3f(start_cell_number);
    let weights = linearSplineWeights(fractional_pos);

    var mass = 0.;

    for (var z = 0u; z < 2; z++) {
        for (var y = 0u; y < 2; y++) {
            for (var x = 0u; x < 2; x++) {
                let cell_number = start_cell_number + vec3u(x, y, z);
                
                if !cellNumberInGridRange(vec3i(cell_number)) { continue; }
                
                let cell_index = linearizeCellIndex(cell_number);
                
                let val = f32(atomicLoad(&mass_grid[cell_index])) / uniforms.fixedPointScale;
                
                let weight = weights[x].x * weights[y].y * weights[z].z;
                
                mass += val * weight;
            }
        }
    }
    
    let cellVolume = cellSize.x * cellSize.y * cellSize.z;
    return mass / cellVolume * 0.000001;
}

fn henyeyGreenstein(ray_light_dot: f32, asymmetry: f32) -> f32 {
    let asymmetry2 = asymmetry * asymmetry;
    let denom = 1 + asymmetry2 - 2 * asymmetry * ray_light_dot;
    return (1 - asymmetry2) / (4 * PI * pow(denom, 1.5));
}

fn doubleHenyeyGreenstein(ray_light_dot: f32, asymmetry_forward: f32, asymmetry_backward: f32) -> f32 {
    let forward_scatter = henyeyGreenstein(ray_light_dot, asymmetry_forward);
    let backward_scatter = henyeyGreenstein(ray_light_dot, -asymmetry_backward);
    return backward_scatter + forward_scatter;
}

fn aabbIntersectionDistances(
    ray_origin: vec3f,
    ray_dir: vec3f,
    grid_min_coords: vec3f,
    grid_max_coords: vec3f,
) -> vec2f {
    let distance_min = (grid_min_coords - ray_origin) / ray_dir;
    let distance_max = (grid_max_coords - ray_origin) / ray_dir;

    let t1 = min(distance_min, distance_max);
    let t2 = max(distance_min, distance_max);
    let t_near = max(max(t1.x, t1.y), t1.z);
    let t_far = min(min(t2.x, t2.y), t2.z);
    return vec2f(t_near, t_far);
}

fn raymarchShadow(pos: vec3f, light_dir: vec3f) -> vec3f {
    var shadow_pos = pos + light_dir * SHADOW_STEP_SIZE;
    var shadow_transmittance = vec3f(1);

    for (var s = 0u; s < N_MAX_SHADOW_STEPS; s++) {
        shadow_pos += light_dir * SHADOW_STEP_SIZE;
        let shadow_density = readDensity(shadow_pos);
        if shadow_density > 1e-4 {
            let shadow_extinction = EXTINCTION_COEFFICIENT * shadow_density;
            shadow_transmittance *= exp(-shadow_extinction * SHADOW_STEP_SIZE * SCATTERING_ALBEDO);
            if all(shadow_transmittance < vec3f(1e-4)) { break; }
        }
    }
    return shadow_transmittance;
}

@compute
@workgroup_size(16, 16)
fn doVolumetricRaymarch(
    @builtin(global_invocation_id) global_id: vec3u,
) {
    let texture_dims = textureDimensions(outputTexture);
    if global_id.x >= texture_dims.x || global_id.y >= texture_dims.y {
        return;
    }

    let uv = vec2f(global_id.xy) / vec2f(texture_dims);
    let ray = calculateViewRay(uv, texture_dims);
    let ray_origin = ray.origin;
    let ray_dir = ray.dir;

    let light_dir = normalize(vec3f(0.2, 0.18, 0.9));
    
    let light_col = vec3f(0.95, 0.995, 1) * 5;
    
    let ambient_col = vec3f(0, 0.1, 0.2);

    let distance_bounds = aabbIntersectionDistances(ray_origin, ray_dir, uniforms.gridMinCoords, uniforms.gridMaxCoords);
    let distance_near = distance_bounds.x;
    let distance_far = distance_bounds.y;
    
    let ground_z = uniforms.gridMinCoords.z;
    var distance_ground = 1e20;
    var hit_ground = false;
    
    if abs(ray_dir.z) > 1e-4 {
        let candidate_distance_ground = (ground_z - ray_origin.z) / ray_dir.z;
        if candidate_distance_ground > 0 {
            let hit_pos = ray_origin + candidate_distance_ground * ray_dir;
            if all(hit_pos.xy >= uniforms.gridMinCoords.xy) && all(hit_pos.xy <= uniforms.gridMaxCoords.xy) {
                distance_ground = candidate_distance_ground;
                hit_ground = true;
            }
        }
    }

    let ray_hits_volume = distance_near <= distance_far && distance_far >= 0;

    if !ray_hits_volume {
        textureStore(outputTexture, global_id.xy, vec4f(0, 0, 0, 0));
        return;
    }


    let volume_start = max(0, distance_near);

    let distance_start = volume_start;
    var distance_end = min(distance_far, distance_ground);
    
    var current_ray_distance = distance_start;

    var transmittance = vec3f(1);
    var out_col = vec3f(0);
    var depth_written = false;
    var recorded_depth = 1e20;

    for (var i = 0u; i < N_MAX_STEPS; i++) {
        if current_ray_distance >= distance_end || all(transmittance < vec3f(0.01)) { break; }

        let pos = ray_origin + current_ray_distance * ray_dir;
        let density = readDensity(pos);

        if density > 0.001 {
            if !depth_written {
                recorded_depth = current_ray_distance;
                depth_written = true;
            }
            
            let local_extinction = EXTINCTION_COEFFICIENT * density; // σ_t
            let local_scattering = local_extinction * SCATTERING_ALBEDO; // σ_s
            
            // exponential assuming a constant density, since every step can be thought of as an independent trial
            // in a geometric probability distribution
            let step_transmittance = exp(-local_extinction * STEP_SIZE * SCATTERING_ALBEDO); // T
            let phase = doubleHenyeyGreenstein(dot(ray_dir, light_dir), 0.5, 0.5); // P
            
            let shadow_transmittance = raymarchShadow(pos, light_dir);

            let light = ambient_col + light_col * phase * shadow_transmittance;
            let light_addition_fac = light * local_scattering * (1 - step_transmittance) / max(local_extinction, 1e-4);
            
            out_col += transmittance * light_addition_fac;
            transmittance *= step_transmittance;
        }

        current_ray_distance += STEP_SIZE;
    }

    if hit_ground && any(transmittance > vec3f(0.01)) {
        let pos = ray_origin + distance_ground * ray_dir;
        let shadow = raymarchShadow(pos, light_dir);
        let ground_albedo = vec3f(0.05);
        let ground_col = ambient_col + ground_albedo * (light_col * shadow * max(0.0, light_dir.z));
        out_col += transmittance * ground_col;
        transmittance = vec3f();

        if !depth_written {
            recorded_depth = distance_ground;
        }
    }

    let alpha = 1 - max(transmittance.r, max(transmittance.g, transmittance.b));
    textureStore(outputTexture, global_id.xy, vec4f(
        pow(out_col.r, 1/2.2),
        pow(out_col.g, 1/2.2),
        pow(out_col.b, 1/2.2),
        1,
    ) * alpha);
    
    textureStore(depthTexture, global_id.xy, vec4f(recorded_depth, 0, 0, 0));
}
