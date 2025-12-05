@group(0) @binding(1) var smoothedDepthTexture: texture_2d<f32>;
@group(0) @binding(2) var normalTexture: texture_2d<f32>;
@group(0) @binding(3) var diffuseOutputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var specularAmbientTexture: texture_storage_2d<rgba8unorm, write>;

const NOISE_SCALE = 10.;
const NOISE_STRENGTH_PACKED = 0.05;
const NOISE_STRENGTH_LOOSE = 0.125;

const LIGHT_DIR = vec3f(0.5, 0.3, 0.8);
const AMBIENT_COLOR = vec3f(0.05, 0.08, 0.1);
const DIFFUSE_COLOR = vec3f(0.65, 0.68, 0.69);
const DIFFUSE_STRENGTH = 0.7;
const SPECULAR_STRENGTH = 0.3;
const SHININESS = 2.;


// Zirr-Kaplanyan glints
// crystal density (number of ice crystals per square meter)
const GLINT_CRYSTAL_DENSITY = 100000.;
// base grid scale for finest LOD level, in world units
const GLINT_BASE_SCALE = 1000.;
// number of LOD levels in hierarchy
const GLINT_LOD_LEVELS = 4u;
// roughness for glint microfacet distribution (smaller = sharper glints)
const GLINT_ROUGHNESS = 0.125;
const GLINT_INTENSITY = 0.75;
// screen-space pixel scale for LOD calculation
const GLINT_PIXEL_SCALE = 0.2;

// GGX/Trowbridge-Reitz NDF
fn D_GGX(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH2 = NdotH * NdotH;
    let denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn hashGridCell(cell: vec3i, lod: u32) -> f32 {
    let seed = vec4u(
        bitcast<u32>(cell.x),
        bitcast<u32>(cell.y),
        bitcast<u32>(cell.z),
        lod,
    );
    return f32(hash4(seed)) / f32(0xFFFFFFFFu);
}

// calculate appropriate grid level based on pixel footprint
fn calculateGlintLOD(pos_world: vec3f, camera_pos: vec3f) -> u32 {
    let dist = length(pos_world - camera_pos);
    // pixel footprint scales with distance
    let footprint = dist * GLINT_PIXEL_SCALE;
    // select LOD level where grid cell size matches footprint
    let lod_float = log2(max(footprint * GLINT_BASE_SCALE, 1.0));
    return clamp(u32(lod_float), 0u, GLINT_LOD_LEVELS - 1u);
}

fn computeMultiscaleGlint(
    pos_world: vec3f,
    normal: vec3f,
    view_dir: vec3f,
    light_dir: vec3f,
    camera_pos: vec3f
) -> f32 {
    // 1. Calculate half-vector for specular reflection
    let H = normalize(light_dir + view_dir);
    let NdotH = max(dot(normal, H), 0.0);
    
    // Skip if facing away from the highlight direction
    if NdotH < 0.001 {
        return 0.0;
    }
    
    // 2. Calculate NDF probability - this is P in the binomial distribution
    let P = D_GGX(NdotH, GLINT_ROUGHNESS);
    
    // 3. Select LOD level based on pixel footprint
    let lod = calculateGlintLOD(pos_world, camera_pos);
    
    // 4. Calculate grid scale for this LOD level
    let grid_scale = GLINT_BASE_SCALE / pow(2.0, f32(lod));
    
    // 5. Find grid cell at this scale
    let grid_pos = pos_world * grid_scale;
    let grid_cell = vec3i(floor(grid_pos));
    
    // 6. Generate stable random value for this grid cell
    let rand = hashGridCell(grid_cell, lod);
    
    // 7. Calculate number of crystals in this grid cell
    let cell_area = 1.0 / (grid_scale * grid_scale);
    let N = GLINT_CRYSTAL_DENSITY * cell_area;
    
    // 8. Statistical model: Binomial distribution B(N, P)
    // Mean: mu = N * P (gives standard specular)
    // Variance: sigma^2 = N * P * (1 - P)
    // When N is small, variance relative to mean is high -> sparkles
    // When N is large, variance relative to mean is low -> smooth highlight
    
    let mean = N * P;
    let variance = N * P * (1.0 - P);
    let std_dev = sqrt(max(variance, 0.0001));
    
    // 9. Threshold test using inverse CDF sampling
    // Map random value to number of reflecting crystals
    // Using approximation: glint occurs when random exceeds threshold
    let relative_variance = std_dev / max(mean, 0.0001);
    
    // Glint probability: higher when variance is high relative to mean
    let glint_threshold = 1.0 - min(relative_variance * P * 10.0, 1.0);
    
    // 10. Apply glint if random value passes threshold
    // Smooth the edge to avoid hard thresholding artifacts
    let glint_raw = smoothstep(glint_threshold, glint_threshold + 0.1, rand);
    
    // 11. Modulate by NdotH for highlight shape
    return glint_raw * NdotH * NdotH;
}

fn noiseGradient3(p: vec3f) -> vec3f {
    let eps = 0.01;
    let n = noise3(p);
    let nx = noise3(p + vec3f(eps, 0, 0));
    let ny = noise3(p + vec3f(0, eps, 0));
    let nz = noise3(p + vec3f(0, 0, eps));
    return vec3f(nx - n, ny - n, nz - n) / eps;
}

fn triplanarNoiseNormal(pos_world: vec3f, normal_world: vec3f) -> vec3f {
    let weights = abs(normal_world);
    let weights_norm = weights / max(dot(weights, vec3f(1)), 0.001);
    
    let noise_xy = noiseGradient3(vec3f(pos_world.xy * NOISE_SCALE, 0));
    let noise_xz = noiseGradient3(vec3f(pos_world.xz * NOISE_SCALE, 0));
    let noise_yz = noiseGradient3(vec3f(pos_world.yz * NOISE_SCALE, 0));
    
    var noise_normal = noise_xy * weights_norm.z + noise_xz * weights_norm.y + noise_yz * weights_norm.x;
    noise_normal = noise_normal - dot(noise_normal, normal_world) * normal_world;
    return noise_normal;
}

fn reconstructWorldPos(coords: vec2i, depth: f32, screen_size: vec2f) -> vec3f {
    let uv = (vec2f(coords) + 0.5) / screen_size;
    let ndc = vec2f(uv.x, 1.0 - uv.y) * 2.0 - 1.0;
    let clip_pos = vec4f(ndc, depth, 1.0);
    let world_pos_hom = uniforms.viewProjInvMat * clip_pos;
    return world_pos_hom.xyz / world_pos_hom.w;
}

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let screen_size = vec2f(textureDimensions(smoothedDepthTexture));
    let coords = vec2i(global_id.xy);
    
    if any(coords >= vec2i(screen_size)) {
        return;
    }
    
    let depth = textureLoad(smoothedDepthTexture, coords, 0).r;
    
    if depth >= 1 {
        // bg pixel - write to both textures
        textureStore(diffuseOutputTexture, coords, vec4f(0.0, 0.0, 0.0, 0.0));
        textureStore(specularAmbientTexture, coords, vec4f(0.0, 0.0, 0.0, 0.0));
        return;
    }
    
    let normal_data = textureLoad(normalTexture, coords, 0);
    var normal_world = normal_data.xyz;
    let compression_volume_fac = normal_data.w;

    
    let normal_len = length(normal_world);
    normal_world = select(
        normal_world / normal_len,
        vec3f(0, 0, 1),
        normal_len < 1e-4,
    );
    

    let pos_world = reconstructWorldPos(coords, depth, screen_size);



    let noise_strength = mix(NOISE_STRENGTH_PACKED, NOISE_STRENGTH_LOOSE, saturate(compression_volume_fac));
    let noise_normal = triplanarNoiseNormal(pos_world, normal_world);    
    var perturbed_normal = normalize(normal_world + noise_strength * noise_normal);
    


    let light_dir_world = normalize(LIGHT_DIR); 
    let view_dir_world = normalize(uniforms.cameraPos - pos_world);
    
    // diffuse
    let diffuse = max(dot(perturbed_normal, light_dir_world), 0);
    
    // Blinn-Phong specular
    let half_dir = normalize(light_dir_world + view_dir_world);
    let specular = pow(max(dot(perturbed_normal, half_dir), 0), SHININESS);
    
    // Zirr-Kaplanyan multiscale procedural glint
    let glint = computeMultiscaleGlint(
        pos_world,
        perturbed_normal,
        view_dir_world,
        light_dir_world,
        uniforms.cameraPos
    );
    let specular_with_glint = /*specular * SPECULAR_STRENGTH +*/ glint * GLINT_INTENSITY;

    let diffuse_color = DIFFUSE_COLOR * diffuse * DIFFUSE_STRENGTH;
    let specular_color = vec3f(specular_with_glint);
    
    textureStore(diffuseOutputTexture, coords, vec4f(diffuse_color, 1));
    
    let base_lighting = AMBIENT_COLOR + diffuse_color + specular_color;
    textureStore(specularAmbientTexture, coords, vec4f(base_lighting, 1));
}
