@group(1) @binding(0) var<storage, read> particleData: array<ParticleData>;

const SPHERE_RADIUS = 0.02;
const SMOOTHNESS = 1.0 / 32.0; // Adjusted for better blending

// Volumetric Parameters
const SIGMA_T = 724.0;
const ALBEDO = vec3f(0.9, 0.95, 1.0);
const G = 0.5;
const SQRT_2 = sqrt(2);
const LIGHT_DIR = vec3f(0.5, 1.0, 0.5) / SQRT_2;
const LIGHT_COLOR = vec3f(2.0, 2.0, 2.0);
const AMBIENT_LIGHT = vec3f(0.2, 0.2, 0.3);

fn smoothMin(a: f32, b: f32) -> f32 {
    // quartic polynomial smooth min from https://iquilezles.org/articles/smin/
    let smoothFac = SMOOTHNESS * 16.0 / 3.0;
    let blendFac = max(0.0, smoothFac - abs(a - b)) / smoothFac;
    return min(a, b) - blendFac * blendFac * blendFac * (4.0 - blendFac) * smoothFac / 16.0;
}

fn allParticlesSdf(pos: vec3f) -> f32 {
    if (arrayLength(&particleData) == 0) { return 100.0; }
    var dist = length(pos - particleData[0].pos) - SPHERE_RADIUS;
    
    // Optimization: Only check particles if we are reasonably close? 
    // For now, keep it simple but maybe limit the loop if performance is bad.
    // A spatial acceleration structure would be ideal but is out of scope for this shader edit.
    for (var i = 1u; i < arrayLength(&particleData); i++) {
        let curDist = length(pos - particleData[i].pos) - SPHERE_RADIUS;
        dist = smoothMin(dist, curDist);
    }
    
    return dist;
}

fn phaseHG(g: f32, cosTheta: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * 3.14159 * pow(denom, 1.5));
}

fn getDensity(pos: vec3f) -> f32 {
    let dist = allParticlesSdf(pos);
    // Map negative distance (inside) to density.
    // We assume density is 1.0 inside the "core" of the snow and falls off.
    // Using a smoother transition for the volume boundary.
    return smoothstep(0.0, -0.01, dist);
}

fn getLightTransmittance(pos: vec3f, lightDir: vec3f) -> f32 {
    var t = 0.01; // Start slightly offset to avoid self-intersection
    var opticalDepth = 0.0;
    const MAX_SHADOW_STEPS = 5u; // Low count for performance
    const SHADOW_STEP_SIZE = 0.02;

    for (var i = 0u; i < MAX_SHADOW_STEPS; i++) {
        let p = pos + lightDir * t;
        let density = getDensity(p);
        if (density > 0.0) {
            opticalDepth += density * SHADOW_STEP_SIZE * SIGMA_T;
        }
        t += SHADOW_STEP_SIZE;
    }
    return exp(-opticalDepth);
}

@fragment
fn frag(
    in: RaymarchVertexOut,
) -> @location(0) vec4f {
    if arrayLength(&particleData) == 0 { return vec4f(0.0, 0.0, 0.0, 1.0); }

    // use inverse view-projection to get ray direction that respects FOV and aspect ratio
    let nearPointHom = uniforms.viewProjInvMat * vec4f(in.uvCentered, 0.0, 1.0);
    let farPointHom = uniforms.viewProjInvMat * vec4f(in.uvCentered, 1.0, 1.0);
    
    let nearPoint = nearPointHom.xyz / nearPointHom.w;
    let farPoint = farPointHom.xyz / farPointHom.w;
    
    let rayDir = normalize(farPoint - nearPoint);
    var rayPos = nearPoint;

    var transmittance = vec3f(1.0);
    var radiance = vec3f(0.0);

    const MAX_STEPS = 64u;
    const STEP_SIZE = 0.01;
    const MAX_DIST = 10.0;

    var t = 0.0;
    
    // Initial march to find the volume
    var hitVolume = false;
    for (var i = 0u; i < 50u; i++) {
        let dist = allParticlesSdf(rayPos);
        if (dist < 0.01) {
            hitVolume = true;
            break;
        }
        t += max(dist, STEP_SIZE);
        rayPos = nearPoint + t * rayDir;
        if (t > MAX_DIST) { break; }
    }

    if (!hitVolume) { return vec4f(0.0, 0.0, 0.0, 1.0); }

    // Volumetric Ray Marching
    for (var i = 0u; i < MAX_STEPS; i++) {
        let density = getDensity(rayPos);

        if (density > 0.001) {
            let extinction = SIGMA_T * density;
            let stepTransmittance = exp(-extinction * STEP_SIZE);
            
            // In-scattering
            // Direct Light
            let lightTransmittance = getLightTransmittance(rayPos, LIGHT_DIR);
            let phase = phaseHG(G, dot(rayDir, LIGHT_DIR));
            let scattering = SIGMA_T * ALBEDO * density * phase;
            
            let incomingLight = LIGHT_COLOR * lightTransmittance + AMBIENT_LIGHT;
            
            // Analytic integration over the step (assuming constant density over step)
            let integScatt = incomingLight * scattering * (1.0 - stepTransmittance) / extinction;
            
            radiance += transmittance * integScatt;
            transmittance *= stepTransmittance;
        }

        if (transmittance.x < 0.01 && transmittance.y < 0.01 && transmittance.z < 0.01) {
            break;
        }

        t += STEP_SIZE;
        rayPos = nearPoint + t * rayDir;
        if (t > MAX_DIST) { break; }
    }

    // Background blending (if needed, currently black)
    let background = vec3f(0.05, 0.05, 0.1); // Dark blueish background
    radiance += transmittance * background;

    return vec4f(radiance, 1.0);
}