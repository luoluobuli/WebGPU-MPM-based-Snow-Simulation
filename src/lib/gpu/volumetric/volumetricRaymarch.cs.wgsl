@group(1) @binding(0) var<storage, read_write> densityGrid: array<atomic<u32>>;
@group(1) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;

// Constants
const PI = 3.1415926;
const DENSITY_SCALE = 1000.0;
const EXTINCTION_COEFFICIENT = 724.;
const SCATTERING_ALBEDO = 0.95;
const HENYEY_GREENSTEIN_ASYMMETRY = 0.5;
const STEP_SIZE = 0.15;
const MAX_STEPS = 256u;

fn readDensity(worldPos: vec3f) -> f32 {
    if any(worldPos < uniforms.gridMinCoords) || any(worldPos >= uniforms.gridMaxCoords) {
        return 0;
    }

    let gridRange = uniforms.gridMaxCoords - uniforms.gridMinCoords;
    let gridRes = vec3f(uniforms.gridResolution);
    let cellSize = gridRange / gridRes;
    
    let localPos = worldPos - uniforms.gridMinCoords;
    let gridPos = localPos / cellSize;
    
    // Trilinear interpolation manually since we are using a storage buffer
    let splatPos = gridPos - 0.5;
    let start_cell_number = vec3u(splatPos);
    let fractional_pos = splatPos - vec3f(start_cell_number);

    var mass = 0.;

    for (var z = 0u; z < 2; z++) {
        for (var y = 0u; y < 2; y++) {
            for (var x = 0u; x < 2; x++) {
                let cell_number = start_cell_number + vec3u(x, y, z);
                
                if !cellNumberInGridRange(vec3i(cell_number)) { continue; }
                
                let cell_index = linearizeCellIndex(cell_number);
                
                let val = f32(atomicLoad(&densityGrid[cell_index])) / DENSITY_SCALE;
                
                let weight = 
                    select(fractional_pos.x, 1 - fractional_pos.x, x == 0) *
                    select(fractional_pos.y, 1 - fractional_pos.y, y == 0) *
                    select(fractional_pos.z, 1 - fractional_pos.z, z == 0);
                
                mass += val * weight;
            }
        }
    }
    
    let cellVolume = cellSize.x * cellSize.y * cellSize.z;
    return mass / cellVolume * 0.00000005;
}

fn henyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * pow(denom, 1.5));
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

    let light_dir = normalize(vec3f(0.5, 0.5, 1.0));
    let light_col = vec3f(1);

    let distance_bounds = aabbIntersectionDistances(ray_origin, ray_dir, uniforms.gridMinCoords, uniforms.gridMaxCoords);
    let distance_near = distance_bounds.x;
    let distance_far = distance_bounds.y;

    if distance_near > distance_far || distance_far < 0 {
        textureStore(outputTexture, global_id.xy, vec4f(0, 0, 0, 0));
        return;
    }

    let distance_start = max(0, distance_near);
    let distance_end = distance_far;
    let jitter = f32(hash2(bitcast<vec2u>(vec2f(global_id.xy) + vec2f(uniforms.simulationTimestep)))) / f32(0xFFFFFFFF); // Time varying jitter
    var current_ray_distance = distance_start + jitter * STEP_SIZE;

    var transmittance = 1.;
    var accumulatedColor = vec3f(0);

    for (var i = 0u; i < MAX_STEPS; i++) {
        if current_ray_distance >= distance_end || transmittance < 0.01 { break; }

        let pos = ray_origin + current_ray_distance * ray_dir;
        let density = readDensity(pos);

        if density > 0.001 {
            let local_extinction = EXTINCTION_COEFFICIENT * density; // σ_t
            let local_scattering = local_extinction * SCATTERING_ALBEDO; // σ_s
            
            let stepTransmittance = exp(-local_extinction * STEP_SIZE); // T
            let phase = henyeyGreenstein(dot(ray_dir, light_dir), HENYEY_GREENSTEIN_ASYMMETRY); // P
            const AMBIENT = vec3f(0.2);
            let scattering = (light_col * phase + AMBIENT) * local_scattering;
            let lightAdded = scattering * (1.0 - stepTransmittance) / max(local_extinction, 0.0001);
            
            accumulatedColor += transmittance * lightAdded;
            transmittance *= stepTransmittance;
        }

        current_ray_distance += STEP_SIZE;
    }


    let alpha = 1 - transmittance;
    textureStore(outputTexture, global_id.xy, vec4f(accumulatedColor, alpha));
}
