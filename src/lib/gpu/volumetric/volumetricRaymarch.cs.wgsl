@group(1) @binding(0) var<storage, read_write> mass_grid: array<atomic<u32>>;
@group(1) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(2) var depthTexture: texture_storage_2d<r32float, write>;

const PI = 3.1415926;
const EXTINCTION_COEFFICIENT = 724.;
const SCATTERING_ALBEDO = 0.95;
const STEP_SIZE = 0.1;
const N_MAX_STEPS = 256u;

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
                
                let val = f32(atomicLoad(&mass_grid[cell_index])) / MASS_FIXED_POINT_SCALE;
                
                let weight = weights[x].x * weights[y].y * weights[z].z;
                
                mass += val * weight;
            }
        }
    }
    
    let cellVolume = cellSize.x * cellSize.y * cellSize.z;
    return mass / cellVolume * 0.000000005;
}

fn henyeyGreenstein(ray_light_dot: f32, asymmetry: f32) -> f32 {
    let asymmetry2 = asymmetry * asymmetry;
    let denom = 1 + asymmetry2 - 2 * asymmetry * ray_light_dot;
    return (1 - asymmetry2) / (4 * PI * pow(denom, 1.5));
}

fn twoLobeHenyeyGreenstein(ray_light_dot: f32, asymmetry_forward: f32, asymmetry_backward: f32) -> f32 {
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

fn raymarchShadow(pos: vec3f, light_dir: vec3f) -> f32 {
    var shadow_pos = pos;
    var shadow_transmittance = 1.;
    const N_MAX_SHADOW_STEPS = 50u;
    const SHADOW_STEP_SIZE = STEP_SIZE;

    for (var s = 0u; s < N_MAX_SHADOW_STEPS; s++) {
        shadow_pos += light_dir * SHADOW_STEP_SIZE;
        let shadow_density = readDensity(shadow_pos);
        if shadow_density > 1e-4 {
            let shadow_extinction = EXTINCTION_COEFFICIENT * shadow_density;
            shadow_transmittance *= exp(-shadow_extinction * SHADOW_STEP_SIZE);
            if shadow_transmittance < 1e-4 { break; }
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
    let uvNormalized = vec2f(uv.x, 1 - uv.y) * 2 - 1;
    
    let nearPosHom = uniforms.viewProjInvMat * vec4f(uvNormalized, 0, 1);
    let nearPos = nearPosHom.xyz / nearPosHom.w;
    
    let farPosHom = uniforms.viewProjInvMat * vec4f(uvNormalized, 1, 1);
    let farPos = farPosHom.xyz / farPosHom.w;

    let ray_origin = nearPos;
    let ray_dir = normalize(farPos - nearPos);

    let light_dir = normalize(vec3f(0.25, 0.5, 1));
    let light_col = vec3f(3.2, 3.8, 4);
    const AMBIENT_LIGHT_COL = vec3f(0.01);

    let distance_bounds = aabbIntersectionDistances(ray_origin, ray_dir, uniforms.gridMinCoords, uniforms.gridMaxCoords);
    let distance_near = distance_bounds.x;
    let distance_far = distance_bounds.y;

    let ray_hits_volume = distance_near <= distance_far && distance_far >= 0;

    // let ground_z = uniforms.gridMinCoords.z;
    // var t_ground = 1e20;
    // var hit_ground = false;
    
    // if (abs(ray_dir.z) > 1e-5) {
    //     let t = (ground_z - ray_origin.z) / ray_dir.z;
    //     if (t > 0.0) {
    //         t_ground = t;
    //         hit_ground = true;
    //     }
    // }

    if !ray_hits_volume {
        textureStore(outputTexture, global_id.xy, vec4f(0, 0, 0, 0));
        return;
    }

    let volume_start = max(0, distance_near);

    let distance_start = volume_start;
    var distance_end = distance_far;
    
    // if (hit_ground && t_ground < distance_end) {
    //     distance_end = t_ground;
    // }
    let jitter = f32(hash2(global_id.xy)) / f32(0xFFFFFFFF);
    var current_ray_distance = distance_start + jitter * STEP_SIZE;

    var transmittance = 1.;
    var out_col = vec3f(0);
    var depth_written = false;
    var recorded_depth = 1e20;

    for (var i = 0u; i < N_MAX_STEPS; i++) {
        if current_ray_distance >= distance_end || transmittance < 0.01 { break; }

        let pos = ray_origin + current_ray_distance * ray_dir;
        let density = readDensity(pos);

        if density > 0.001 {
            // Record depth at first significant hit
            if !depth_written {
                recorded_depth = current_ray_distance;
                depth_written = true;
            }
            
            let local_extinction = EXTINCTION_COEFFICIENT * density; // σ_t
            let local_scattering = local_extinction * SCATTERING_ALBEDO; // σ_s
            
            // exponential assuming a constant density, since every step can be thought of as an independent trial
            // in a geometric probability distribution
            let stepTransmittance = exp(-local_extinction * STEP_SIZE); // T
            let phase = twoLobeHenyeyGreenstein(dot(ray_dir, light_dir), 0.5, 0.2); // P
            
            let shadow_transmittance = raymarchShadow(pos, light_dir);

            let light = AMBIENT_LIGHT_COL + light_col * phase * shadow_transmittance;
            let light_addition_fac = light * local_scattering * (1 - stepTransmittance) / max(local_extinction, 0.0001);
            
            out_col += transmittance * light_addition_fac;
            transmittance *= stepTransmittance;
        }

        current_ray_distance += STEP_SIZE;
    }


    // if (hit_ground && transmittance > 0) {
    //     let pos = ray_origin + t_ground * ray_dir;
    //     let shadow = raymarchShadow(pos, light_dir);
    //     let ground_albedo = vec3f(0.2);
    //     let ground_col = ground_albedo * (AMBIENT_LIGHT_COL + light_col * shadow * max(0, light_dir.z));
    //     out_col += transmittance * ground_col;
    //     transmittance = 0;
    // }

    let alpha = 1 - transmittance;
    textureStore(outputTexture, global_id.xy, vec4f(
        pow(out_col.r, 1/2.2),
        pow(out_col.g, 1/2.2),
        pow(out_col.b, 1/2.2),
        alpha,
    ));
    
    textureStore(depthTexture, global_id.xy, vec4f(recorded_depth, 0, 0, 0));
}
