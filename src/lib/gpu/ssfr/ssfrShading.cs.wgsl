// SSFR Shading pass with granular noise injection
// Perturbs normals with triplanar noise and applies lighting

@group(0) @binding(1) var smoothedDepthTexture: texture_2d<f32>;
@group(0) @binding(2) var normalTexture: texture_2d<f32>;
@group(0) @binding(3) var outputTexture: texture_storage_2d<rgba8unorm, write>;

const NOISE_SCALE: f32 = 20.0;
const NOISE_STRENGTH_PACKED: f32 = 0.05;
const NOISE_STRENGTH_LOOSE: f32 = 0.25;

const LIGHT_DIR: vec3f = vec3f(0.5, 0.3, 0.8);
const AMBIENT: f32 = 0.15;
const DIFFUSE_STRENGTH: f32 = 0.7;
const SPECULAR_STRENGTH: f32 = 0.3;
const SHININESS: f32 = 32.0;

const PARTICLE_COLOR: vec3f = vec3f(0.95, 0.97, 1);

fn noiseGradient3(p: vec3f) -> vec3f {
    let eps = 0.01;
    let n = noise3(p);
    let nx = noise3(p + vec3f(eps, 0.0, 0.0));
    let ny = noise3(p + vec3f(0.0, eps, 0.0));
    let nz = noise3(p + vec3f(0.0, 0.0, eps));
    return vec3f(nx - n, ny - n, nz - n) / eps;
}

fn triplanarNoiseNormal(pos_world: vec3f, normal_world: vec3f) -> vec3f {
    let weights = abs(normal_world);
    let weights_norm = weights / max(dot(weights, vec3f(1)), 0.001);
    
    let noise_xy = noiseGradient3(vec3f(pos_world.xy * NOISE_SCALE, 0.0));
    let noise_xz = noiseGradient3(vec3f(pos_world.xz * NOISE_SCALE, 100.0));
    let noise_yz = noiseGradient3(vec3f(pos_world.yz * NOISE_SCALE, 200.0));
    
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
        // bg pixel
        textureStore(outputTexture, coords, vec4f(0.0, 0.0, 0.0, 0.0));
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
    let diffuse = max(dot(perturbed_normal, light_dir_world), 0.0);
    
    // Blinn-Phong specular
    let half_dir = normalize(light_dir_world + view_dir_world);
    let specular = pow(max(dot(perturbed_normal, half_dir), 0.0), SHININESS);
    

    let ambient_color = PARTICLE_COLOR * AMBIENT;
    let diffuse_color = PARTICLE_COLOR * diffuse * DIFFUSE_STRENGTH;
    let specular_color = specular * SPECULAR_STRENGTH;
    
    var final_color = ambient_color + diffuse_color + specular_color;
    final_color = pow(final_color, vec3f(1 / 2.2));
    
    textureStore(outputTexture, coords, vec4f(final_color, 1));
}
