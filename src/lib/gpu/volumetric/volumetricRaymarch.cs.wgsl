@group(1) @binding(0) var<storage, read> densityGrid: array<atomic<u32>>;
@group(1) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;

// Constants
const PI: f32 = 3.14159265359;
const DENSITY_SCALE: f32 = 1000.0;
const SIGMA_T: f32 = 724.0; // Extinction coefficient (m^-1)
const ALBEDO: f32 = 0.95; // Scattering albedo
const G: f32 = 0.5; // Henyey-Greenstein asymmetry
const STEP_SIZE: f32 = 0.02; // 2cm step size (tunable)
const MAX_STEPS: u32 = 256; // Max steps to prevent TDR

// Helper to get density from grid
fn getDensity(worldPos: vec3f) -> f32 {
    if (any(worldPos < uniforms.gridMinCoords) || any(worldPos >= uniforms.gridMaxCoords)) {
        return 0.0;
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

    var density: f32 = 0.0;

    for (var z = 0u; z < 2u; z++) {
        for (var y = 0u; y < 2u; y++) {
            for (var x = 0u; x < 2u; x++) {
                let neighborIndex = baseIndex + vec3u(x, y, z);
                
                if (any(neighborIndex >= uniforms.gridResolution)) {
                    continue;
                }

                let idx = neighborIndex.x + 
                          neighborIndex.y * uniforms.gridResolution.x + 
                          neighborIndex.z * uniforms.gridResolution.x * uniforms.gridResolution.y;
                
                let val = f32(atomicLoad(&densityGrid[idx])) / DENSITY_SCALE;
                
                let weight = 
                    select(1.0 - w.x, w.x, x == 0u) *
                    select(1.0 - w.y, w.y, y == 0u) *
                    select(1.0 - w.z, w.z, z == 0u);
                
                density += val * weight;
            }
        }
    }
    
    // The stored density is "Mass". We need "Density" = Mass / Volume.
    // Cell volume = cellSize.x * cellSize.y * cellSize.z
    let cellVolume = cellSize.x * cellSize.y * cellSize.z;
    return density / cellVolume; 
    // Note: This might be too high if particles are dense. We might need a tuning factor.
    // Let's assume the stored value is already proportional to density for now, or tune later.
    // Actually, let's just return density * 0.01 for now to avoid whiteout.
}

fn henyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * pow(denom, 1.5));
}

// Simple random for jitter
fn random(st: vec2f) -> f32 {
    return fract(sin(dot(st.xy, vec2f(12.9898, 78.233))) * 43758.5453123);
}

@compute @workgroup_size(16, 16)
fn doVolumetricRaymarch(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(outputTexture);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let uv = vec2f(global_id.xy) / vec2f(dims);
    
    // Ray Generation
    // Convert UV to NDC (-1 to 1)
    let ndc = vec2f(uv.x, 1.0 - uv.y) * 2.0 - 1.0;
    
    // Unproject Near Plane (z = 0.0)
    let nearPosHom = uniforms.viewProjInvMat * vec4f(ndc, 0.0, 1.0);
    let nearPos = nearPosHom.xyz / nearPosHom.w;

    // Unproject Far Plane (z = 1.0)
    let farPosHom = uniforms.viewProjInvMat * vec4f(ndc, 1.0, 1.0);
    let farPos = farPosHom.xyz / farPosHom.w;
    
    let rayOrigin = nearPos;
    let rayDir = normalize(farPos - nearPos);

    let lightDir = normalize(vec3f(0.5, 0.5, 1.0));
    let lightColor = vec3f(1.0); // White sun

    // Intersection with Volume Bounds (AABB)
    // We only march inside the grid bounds.
    let tMin = (uniforms.gridMinCoords - rayOrigin) / rayDir;
    let tMax = (uniforms.gridMaxCoords - rayOrigin) / rayDir;
    let t1 = min(tMin, tMax);
    let t2 = max(tMin, tMax);
    let tNear = max(max(t1.x, t1.y), t1.z);
    let tFar = min(min(t2.x, t2.y), t2.z);

    if (tNear > tFar || tFar < 0.0) {
        // Missed volume - Debug Blue
        textureStore(outputTexture, global_id.xy, vec4f(0.0, 0.0, 0.2, 1.0));
        return;
    }

    let tStart = max(0.0, tNear);
    let tEnd = tFar;

    // Jitter
    let jitter = random(vec2f(global_id.xy) + vec2f(uniforms.simulationTimestep)); // Time varying jitter
    var t = tStart + jitter * STEP_SIZE;

    var transmittance = 1.0;
    var accumulatedColor = vec3f(0.0);

    for (var i = 0u; i < MAX_STEPS; i++) {
        if (t >= tEnd || transmittance < 0.01) {
            break;
        }

        let pos = rayOrigin + t * rayDir;
        let density = getDensity(pos); // * 0.001; // Scale down for visualization if needed

        if (density > 0.001) {
            let sigma_t = SIGMA_T * density; // Local extinction
            let sigma_s = sigma_t * ALBEDO;
            
            // Transmittance for this step
            let stepTransmittance = exp(-sigma_t * STEP_SIZE);
            
            // In-scattering
            // 1. Direct Light (Single Scattering)
            // Shadow ray? Too expensive. Assume constant ambient + directional attenuation?
            // Let's do a simple directional light with shadow approximation or just phase function.
            // For now: Phase function * Light Color * Density
            // To do shadows, we'd need to march towards light. 
            // Let's skip shadow march for "Realtime" first pass, or do a very short one.
            
            let phase = henyeyGreenstein(dot(rayDir, lightDir), G);
            
            // Ambient / Multiple Scattering approximation
            // Just add a base ambient term proportional to density
            let ambient = vec3f(0.2) * density;
            
            let scattering = (lightColor * phase + ambient) * sigma_s;
            
            // Integrate
            // L_in = scattering * step_size (approx)
            // L_out += T * L_in
            // Better integration:
            // S = scattering
            // L_added = S * (1 - stepTransmittance) / sigma_t
            
            let lightAdded = scattering * (1.0 - stepTransmittance) / max(sigma_t, 0.0001);
            
            accumulatedColor += transmittance * lightAdded;
            transmittance *= stepTransmittance;
        }

        t += STEP_SIZE;
    }

    // Alpha blending with background (premultiplied alpha)
    let alpha = 1.0 - transmittance;
    textureStore(outputTexture, global_id.xy, vec4f(accumulatedColor, alpha));
}
