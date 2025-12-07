// Marching cubes screen-space shading
// 5-step snow shader workflow:
// 1. High roughness base for powder look
// 2. Basic displacement noise for surface unevenness
// 3. Detailed high-frequency noise for breakup
// 4. Subsurface scattering with snow-like parameters
// 5. Light glints (secondary specular coat) on peaks
// + Density grid shadow raymarch for volumetric shadows
// + Ground plane with shadows

struct MCParams {
    mcGridRes: vec3u,
    downsampleFactor: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var depthTexture: texture_depth_2d;
@group(0) @binding(2) var normalTexture: texture_2d<f32>;
@group(0) @binding(3) var shadedOutput: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<storage, read> densityGrid: array<u32>;
@group(0) @binding(5) var<uniform> mcParams: MCParams;

const LIGHT_DIR = vec3f(0.6, 0.4, 1);
const AMBIENT_COLOR = vec3f(0.15, 0.22, 0.28);

// high roughness for powdery look
const BASE_ROUGHNESS = 0.95;
const SPECULAR_ROUGHNESS = 0.8;

const SSS_COLOR = vec3f(0.8, 0.85, 0.9);
const SSS_RADIUS = vec3f(0.36, 0.46, 0.60);
const SSS_STRENGTH = 0.5;

const COAT_COLOR = vec3f(1.0, 1.0, 1.0);
const COAT_ROUGHNESS = 0.1;
const COAT_IOR = 1.3;

// Noise scales for displacement
const NOISE_SCALE_BASIC = 8.;
const NOISE_SCALE_DETAIL = 196.;
const NOISE_SCALE_GLINTS = 32.;
const NOISE_STRENGTH_BASIC = 1.5;
const NOISE_STRENGTH_DETAIL = 0.3;

// Shadow raymarch parameters
const N_SHADOW_STEPS = 32u;
const EXTINCTION_COEFFICIENT = 8.;

fn hash31(p: vec3f) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash33(p: vec3f) -> vec3f {
    var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

fn gradientNoise(p: vec3f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    
    return mix(
        mix(
            mix(hash31(i + vec3f(0,0,0)), hash31(i + vec3f(1,0,0)), u.x),
            mix(hash31(i + vec3f(0,1,0)), hash31(i + vec3f(1,1,0)), u.x),
            u.y
        ),
        mix(
            mix(hash31(i + vec3f(0,0,1)), hash31(i + vec3f(1,0,1)), u.x),
            mix(hash31(i + vec3f(0,1,1)), hash31(i + vec3f(1,1,1)), u.x),
            u.y
        ),
        u.z
    ) * 2 - 1;
}

fn fbmNoise(p: vec3f, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;
    
    for (var i = 0; i < octaves; i++) {
        value += amplitude * gradientNoise(pos * frequency);
        frequency *= 2;
        amplitude *= 0.5;
    }
    return value;
}

// for normal preturbation
fn noiseGradient(p: vec3f, scale: f32, octaves: i32) -> vec3f {
    const EPSILON = 1e-6;
    let dx = fbmNoise((p + vec3f(EPSILON, 0, 0)) * scale, octaves) - 
             fbmNoise((p - vec3f(EPSILON, 0, 0)) * scale, octaves);
    let dy = fbmNoise((p + vec3f(0, EPSILON, 0)) * scale, octaves) - 
             fbmNoise((p - vec3f(0, EPSILON, 0)) * scale, octaves);
    let dz = fbmNoise((p + vec3f(0, 0, EPSILON)) * scale, octaves) - 
             fbmNoise((p - vec3f(0, 0, EPSILON)) * scale, octaves);
    return vec3f(dx, dy, dz) / (2 * EPSILON);
}

// brdf
fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn distributionGGX(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometrySmith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

// subsurface
fn subsurfaceScattering(N: vec3f, V: vec3f, L: vec3f, thickness: f32) -> vec3f {
    // Back-lighting / translucency - stronger effect
    let VdotL = max(dot(-V, L), 0.0);
    let translucency = pow(VdotL, 1.5) * 0.8;
    
    // View-dependent wrap lighting for soft diffusion
    let NdotL = dot(N, L);
    let wrapDiffuse = max((NdotL + 0.7) / 1.8, 0.0);
    let forwardScatter = pow(max(dot(V, L), 0.0), 3.0) * 0.4;
    let attenuation = exp(-vec3f(0.5) / SSS_RADIUS * thickness);
    
    // Combine effects with stronger base
    let sss = SSS_COLOR * (translucency + wrapDiffuse * 0.5 + forwardScatter) * attenuation;
    
    return sss * SSS_STRENGTH;
}

// glints
fn glintMask(worldPos: vec3f, N: vec3f, L: vec3f, V: vec3f) -> f32 {
    let noiseVal = fbmNoise(worldPos * NOISE_SCALE_GLINTS, 2);
    
    // Reflection-based: glints appear where view reflects light
    let H = normalize(L + V);
    
    // Combine noise threshold with specular alignment
    // Noise creates sparse distribution, NdotH makes them view-dependent
    const GLINT_THRESHOLD = 0.35;
    let sparkleNoise = smoothstep(GLINT_THRESHOLD, GLINT_THRESHOLD + 0.01, noiseVal);
    
    // Sharp specular falloff for tight sparkles
    let specularFactor = pow((dot(N, H) + 1) * 0.5, 16);
    
    return sparkleNoise * specularFactor;
}

fn coatSpecular(N: vec3f, V: vec3f, L: vec3f, glintStrength: f32) -> vec3f {
    // Simple and bright - just return white sparkles
    return COAT_COLOR * glintStrength;
}

// world pos
fn reconstructWorldPos(coords: vec2i, depth: f32, screenSize: vec2f) -> vec3f {
    let uv = (vec2f(coords) + 0.5) / screenSize;
    let ndc = vec2f(uv.x, 1.0 - uv.y) * 2.0 - 1.0;
    let clipPos = vec4f(ndc, depth, 1.0);
    let worldPosHom = uniforms.viewProjInvMat * clipPos;
    return worldPosHom.xyz / worldPosHom.w;
}

fn worldToScreen(worldPos: vec3f, screenSize: vec2f) -> vec3f {
    let clipPos = uniforms.viewProjMat * vec4f(worldPos, 1.0);
    let ndc = clipPos.xyz / clipPos.w;
    let uv = ndc.xy * 0.5 + 0.5;
    return vec3f(uv.x * screenSize.x, (1.0 - uv.y) * screenSize.y, ndc.z * 0.5 + 0.5);
}

// sampling
fn sampleDensity(worldPos: vec3f) -> f32 {
    // Check bounds
    if (any(worldPos < uniforms.gridMinCoords) || any(worldPos >= uniforms.gridMaxCoords)) {
        return 0.0;
    }
    
    let gridRange = uniforms.gridMaxCoords - uniforms.gridMinCoords;
    let gridRes = vec3f(mcParams.mcGridRes);
    let cellSize = gridRange / gridRes;
    
    let posFromMin = worldPos - uniforms.gridMinCoords;
    let cellCoord = posFromMin / cellSize;
    let cellIndex = vec3i(floor(cellCoord));
    
    // Bounds check
    if (any(cellIndex < vec3i(0)) || any(cellIndex >= vec3i(mcParams.mcGridRes))) {
        return 0.0;
    }
    
    let idx = cellIndex.x + cellIndex.y * i32(mcParams.mcGridRes.x) + 
              cellIndex.z * i32(mcParams.mcGridRes.x * mcParams.mcGridRes.y);
    
    return f32(densityGrid[idx]) / uniforms.fixedPointScale;
}

// shadow raymarch
fn raymarchShadow(worldPos: vec3f, lightDir: vec3f) -> vec3f {
    let gridRange = uniforms.gridMaxCoords - uniforms.gridMinCoords;
    let maxDist = length(gridRange);
    // let stepSize = maxDist / f32(N_SHADOW_STEPS);
    let stepSize = 0.02;
    
    var shadow = vec3f(1);
    var rayPos = worldPos + lightDir * stepSize;  // Start slightly offset
    
    for (var i = 0u; i < N_SHADOW_STEPS; i++) {
        let density = sampleDensity(rayPos);
        
        // Accumulate shadow based on density
        shadow *= exp(-density * EXTINCTION_COEFFICIENT * stepSize * SSS_COLOR);
        
        // Early exit if fully shadowed
        if all(shadow < vec3f(0.01)) {
            return vec3f(0);
        }
        
        rayPos += lightDir * stepSize;
    }
    
    return clamp(shadow, vec3f(0), vec3f(1));
}

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let screenSize = vec2f(textureDimensions(depthTexture));
    let coords = vec2i(global_id.xy);
    
    if any(coords >= vec2i(screenSize)) {
        return;
    }
    
    let depth = textureLoad(depthTexture, coords, 0);
    let lightDir = normalize(LIGHT_DIR);
    
    // Calculate view ray for ground plane
    let uv = (vec2f(coords) + 0.5) / screenSize;
    let ndc = vec2f(uv.x, 1.0 - uv.y) * 2.0 - 1.0;
    let clipNear = vec4f(ndc, 0.0, 1.0);
    let clipFar = vec4f(ndc, 1.0, 1.0);
    let worldNear = uniforms.viewProjInvMat * clipNear;
    let worldFar = uniforms.viewProjInvMat * clipFar;
    let rayOrigin = worldNear.xyz / worldNear.w;
    let rayEnd = worldFar.xyz / worldFar.w;
    let rayDir = normalize(rayEnd - rayOrigin);
    let viewDir = -rayDir;
    
    // No snow and no ground - transparent
    if (depth >= 1.0) {
        textureStore(shadedOutput, coords, vec4f(0.0, 0.0, 0.0, 0.0));
        return;
    }
    
    // Load and normalize base normal
    let normalData = textureLoad(normalTexture, coords, 0);
    var normal = normalData.xyz;
    let normalLen = length(normal);
    
    if normalLen < 0.0001 {
        normal = vec3f(0.0, 0.0, 1.0);
    } else {
        normal = normal / normalLen;
    }
    
    let worldPos = reconstructWorldPos(coords, depth, screenSize);
    
    // ===== DENSITY GRID SHADOW RAYMARCH =====
    let shadow = raymarchShadow(worldPos, lightDir);
    let shadowFactor = shadow;
    
    // ===== STEP 2 & 3: Normal perturbation from displacement noise =====
    // Basic displacement (low frequency for unevenness)
    let basicGrad = noiseGradient(worldPos, NOISE_SCALE_BASIC, 4);
    // Detailed displacement (high frequency for breakup)
    let detailGrad = noiseGradient(worldPos, NOISE_SCALE_DETAIL, 2);
    
    // Perturb normal using noise gradients
    var perturbedNormal = normal;
    perturbedNormal -= basicGrad * NOISE_STRENGTH_BASIC;
    perturbedNormal -= detailGrad * NOISE_STRENGTH_DETAIL;
    perturbedNormal = normalize(perturbedNormal);
    
    // === KEY INSIGHT ===
    // Use BLENDED normal for diffuse (smooth + subtle detail)
    // Use FULL perturbed normal for specular/glints (shows detail)
    let diffuseNormal = normalize(mix(normal, perturbedNormal, 0.2));
    let specularNormal = perturbedNormal;
    
    let V = viewDir;
    let L = lightDir;
    
    // ===== STEP 1: Base diffuse with SMOOTH blended normal =====
    // This prevents harsh splotchy shadows
    let diffuseNdotL = max(dot(diffuseNormal, L), 0.0);
    let wrapDiffuse = max((dot(diffuseNormal, L) + 0.6) / 2.5, 0.0);  // More wrap for softness
    let diffuse = vec3f(0.92, 0.94, 0.98) * wrapDiffuse * shadowFactor;
    
    // Very soft specular with perturbed normal (keeps detail visible)
    let H = normalize(L + V);
    let NdotH = max(dot(specularNormal, H), 0.0);
    let NdotV = max(dot(specularNormal, V), 0.0);
    let NdotL = max(dot(specularNormal, L), 0.0);
    
    let F0 = vec3f(0.02);
    let F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    let D = distributionGGX(NdotH, SPECULAR_ROUGHNESS);
    let G = geometrySmith(NdotV, NdotL, SPECULAR_ROUGHNESS);
    let baseSpecular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001) * NdotL * 0.3 * shadowFactor;

    // ===== STEP 4: Subsurface scattering with SMOOTH normal =====
    // Stronger SSS to fill in remaining shadows
    let thickness = 0.3;  // Thinner for more scattering
    let sss = subsurfaceScattering(normal, V, L, thickness) * (0.2 + 0.8 * shadowFactor);
    
    // ===== STEP 5: Light glints with PERTURBED normal =====
    let glint = glintMask(worldPos, specularNormal, L, V);
    let glintSpec = coatSpecular(specularNormal, V, L, glint) * (0.5 + 0.5 * shadowFactor);
    
    // ===== Fresnel rim lighting =====
    let fresnel = pow(1 - max(dot(diffuseNormal, V), 0), 2.5);
    let rim = fresnel * 0.35 * vec3f(0.9, 0.95, 1) * shadowFactor;
    
    var color = AMBIENT_COLOR + diffuse * 0.9 + baseSpecular + sss + glintSpec + rim;
    
    // // Blend with ground if both visible
    // if (hitGround) {
    //     let groundColor = shadeGround(groundWorldPos, lightDir, viewDir);
    //     // Ground is behind snow, no blend needed
    // }
    
    // tone mapping
    // color = color / (color + vec3f(1));
    
    color = pow(color, vec3f(1 / 2.2));

    textureStore(shadedOutput, coords, vec4f(color, 1));
}
