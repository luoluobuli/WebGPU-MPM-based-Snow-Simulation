@group(1) @binding(0) var<storage, read_write> densityGrid: array<atomic<u32>>;
@group(1) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;

// Constants
const PI: f32 = 3.14159265359;
const DENSITY_SCALE: f32 = 1000.0;
const SIGMA_T: f32 = 724.0; // Extinction coefficient (m^-1)
const ALBEDO: f32 = 0.95; // Scattering albedo
const HENYEY_GREENSTEIN_ASYMMETRY: f32 = 0.5;
const STEP_SIZE: f32 = 0.2; // 2cm step size (tunable)
const MAX_STEPS: u32 = 256; // Max steps to prevent TDR

// Helper to get density from grid
fn getDensity(worldPos: vec3f) -> f32 {
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
    let baseIndex = vec3u(floor(splatPos));
    let w = fract(splatPos);

    var density = 0.;

    for (var z = 0u; z < 2; z++) {
        for (var y = 0u; y < 2; y++) {
            for (var x = 0u; x < 2; x++) {
                let neighborIndex = baseIndex + vec3u(x, y, z);
                
                if any(neighborIndex >= uniforms.gridResolution) {
                    continue;
                }

                let idx = neighborIndex.x + 
                          neighborIndex.y * uniforms.gridResolution.x + 
                          neighborIndex.z * uniforms.gridResolution.x * uniforms.gridResolution.y;
                
                let val = f32(atomicLoad(&densityGrid[idx])) / DENSITY_SCALE;
                
                let weight = 
                    select(w.x, 1 - w.x, x == 0u) *
                    select(w.y, 1 - w.y, y == 0u) *
                    select(w.z, 1 - w.z, z == 0u);
                
                density += val * weight;
            }
        }
    }
    
    // The stored density is "Mass". We need "Density" = Mass / Volume.
    // Cell volume = cellSize.x * cellSize.y * cellSize.z
    let cellVolume = cellSize.x * cellSize.y * cellSize.z;
    return density / cellVolume * 0.00001; 
}

fn henyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * pow(denom, 1.5));
}

fn aabbIntersectionDistances(
    rayOrigin: vec3f,
    rayDir: vec3f,
    minCoords: vec3f,
    maxCoords: vec3f,
) -> vec2f {
    let tMin = (minCoords - rayOrigin) / rayDir;
    let tMax = (maxCoords - rayOrigin) / rayDir;
    let t1 = min(tMin, tMax);
    let t2 = max(tMin, tMax);
    let tNear = max(max(t1.x, t1.y), t1.z);
    let tFar = min(min(t2.x, t2.y), t2.z);
    return vec2f(tNear, tFar);
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

    let rayOrigin = nearPos;
    let rayDir = normalize(farPos - nearPos);

    let lightDir = normalize(vec3f(0.5, 0.5, 1.0));
    let light_col = vec3f(1);

    let distance_bounds = aabbIntersectionDistances(rayOrigin, rayDir, uniforms.gridMinCoords, uniforms.gridMaxCoords);
    let distance_near = distance_bounds.x;
    let distance_far = distance_bounds.y;

    if distance_near > distance_far || distance_far < 0 {
        textureStore(outputTexture, global_id.xy, vec4f(0, 0, 0, 1));
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

        let pos = rayOrigin + current_ray_distance * rayDir;
        let density = getDensity(pos);

        if density > 0.001 {
            let local_extinction = SIGMA_T * density; // σ_t
            let local_scattering = local_extinction * ALBEDO; // σ_s
            
            let stepTransmittance = exp(-local_extinction * STEP_SIZE); // T
            let phase = henyeyGreenstein(dot(rayDir, lightDir), HENYEY_GREENSTEIN_ASYMMETRY); // P
            let ambient = vec3f(0.2) * density;
            let scattering = (light_col * phase + ambient) * local_scattering;
            let lightAdded = scattering * (1.0 - stepTransmittance) / max(local_extinction, 0.0001);
            
            accumulatedColor += transmittance * lightAdded;
            transmittance *= stepTransmittance;
        }

        current_ray_distance += STEP_SIZE;
    }


    // TODO alpha handling is incorrect
    let alpha = 1 - transmittance;
    textureStore(outputTexture, global_id.xy, vec4f(accumulatedColor, alpha));
}
