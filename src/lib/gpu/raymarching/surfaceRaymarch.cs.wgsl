@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> mass_grid: array<atomic<u32>>;
@group(1) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(2) var depthTexture: texture_storage_2d<r32float, write>;
@group(1) @binding(3) var environmentTexture: texture_2d<f32>;
@group(1) @binding(4) var environmentSampler: sampler;

const DENSITY_THRESHOLD = 900.;
const STEP_SIZE = 0.025;
const N_MAX_STEPS = 512u;
const SHADOW_STEP_SIZE = STEP_SIZE * 2.0;

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
                workgroupBarrier(); // Should strictly not be needed if read-only but keeps sync

                let cell_number = start_cell_number + vec3u(x, y, z);
                
                if !cellNumberInGridRange(vec3i(cell_number)) { continue; }
                
                let cell_index = linearizeCellIndex(cell_number);
                
                let val = f32(atomicLoad(&mass_grid[cell_index])) / uniforms.fixedPointScale;
                
                let weight = weights[x].x * weights[y].y * weights[z].z;
                
                mass += val * weight;
            }
        }
    }
    
    workgroupBarrier();
    
    let cellVolume = cellSize.x * cellSize.y * cellSize.z;
    return mass / cellVolume;
}

fn calculateNormal(pos: vec3f) -> vec3f {
    let e = vec3f(0.01, 0.0, 0.0);
    let dx = readDensity(pos + e.xyy) - readDensity(pos - e.xyy);
    let dy = readDensity(pos + e.yxy) - readDensity(pos - e.yxy);
    let dz = readDensity(pos + e.yyx) - readDensity(pos - e.yyx);
    return normalize(-vec3f(dx, dy, dz));
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
    let jitter = f32(hash4(vec4u(bitcast<vec3u>(pos), uniforms.time))) / f32(0xFFFFFFFF);
    var shadow_pos = pos + light_dir * SHADOW_STEP_SIZE * (1.0 + jitter);
    
    for (var s = 0u; s < 32u; s++) {
        let density = readDensity(shadow_pos);
        if density > DENSITY_THRESHOLD {
            return 0.0;
        }
        shadow_pos += light_dir * SHADOW_STEP_SIZE;
        if any(shadow_pos < uniforms.gridMinCoords) || any(shadow_pos >= uniforms.gridMaxCoords) {
            break;
        }
    }
    return 1.0;
}

@compute
@workgroup_size(16, 16)
fn doSurfaceRaymarch(
    @builtin(global_invocation_id) global_id: vec3u,
) {
    let texture_dims = textureDimensions(outputTexture);
    if global_id.x >= texture_dims.x || global_id.y >= texture_dims.y { return; }

    let uv = vec2f(global_id.xy) / vec2f(texture_dims);
    let ray = calculateViewRay(uv, texture_dims);
    let ray_origin = ray.origin;
    let ray_dir = ray.dir;

    let light_dir = normalize(vec3f(0.2, 0.5, 0.8));
    let light_col = vec3f(1.0, 0.98, 0.95);
    let ambient_col = vec3f(0.1, 0.15, 0.25);
    let snow_albedo = vec3f(0.95);

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

    workgroupBarrier();

    let ray_hits_volume = distance_near <= distance_far && distance_far >= 0;
    let volume_start = max(0, distance_near);
    let distance_start = volume_start;
    var distance_end = min(distance_far, distance_ground);
    
    if !ray_hits_volume {
        distance_end = -1.0; // Skip volume loop
    }

    let jitter = f32(hash3(vec3u(global_id.xy, uniforms.time))) / f32(0xFFFFFFFF);
    var current_ray_distance = distance_start + jitter * STEP_SIZE;
    
    var hit_surface = false;
    var final_col = vec4f(0.0);
    var recorded_depth = 1e20;

    // Raymarch volume
    if (ray_hits_volume) {
        for (var i = 0u; i < N_MAX_STEPS; i++) {
            if current_ray_distance >= distance_end { break; }

            let pos = ray_origin + current_ray_distance * ray_dir;
            let density = readDensity(pos);

            if density > DENSITY_THRESHOLD {
                hit_surface = true;
                
                // Refine intersection (binary search could be better, but linear backstep is simple)
                // Let's just use current pos for now.
                
                let normal = calculateNormal(pos);
                let shadow = raymarchShadow(pos + normal * SHADOW_STEP_SIZE, light_dir);
                
                let NdotL = max(dot(normal, light_dir), 0.0);
                let diffuse = light_col * NdotL * shadow;
                let ambient = ambient_col;
                
                let col = snow_albedo * (diffuse + ambient);
                
                final_col = vec4f(col, 1.0);
                recorded_depth = current_ray_distance;
                break;
            }

            current_ray_distance += STEP_SIZE;
        }
    }

    // Check ground if missed volume
    workgroupBarrier();

    if !hit_surface && hit_ground {
        let pos = ray_origin + distance_ground * ray_dir;
        let shadow = raymarchShadow(pos, light_dir);
        let ground_albedo = vec3f(0.05);
        let ground_col = ambient_col + ground_albedo * (light_col * shadow * max(0.0, light_dir.z));
        
        // Simple fog
        let fog_dist = distance_ground;
        let fog_amount = 1.0 - exp(-fog_dist * 0.05);
        let fog_col = vec3f(0.6, 0.7, 0.8);
        
        final_col = vec4f(mix(ground_col, fog_col, fog_amount), 1.0);
        recorded_depth = distance_ground;
    }

    workgroupBarrier();

    textureStore(outputTexture, global_id.xy, vec4f(
        pow(final_col.rgb, vec3f(1.0/2.2)),
        final_col.a
    ));
    
    textureStore(depthTexture, global_id.xy, vec4f(recorded_depth, 0, 0, 0));
}
