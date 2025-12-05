@group(0) @binding(1) var blurredDiffuseTexture: texture_2d<f32>;
@group(0) @binding(2) var baseLightingTexture: texture_2d<f32>;
@group(0) @binding(3) var smoothedDepthTexture: texture_2d<f32>;
@group(0) @binding(4) var outputTexture: texture_storage_2d<rgba8unorm, write>;

const SSS_STRENGTH = 0.5;

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let screen_size = vec2i(textureDimensions(blurredDiffuseTexture));
    let coords = vec2i(global_id.xy);
    
    if any(coords >= screen_size) { return; }
    
    let depth = textureLoad(smoothedDepthTexture, coords, 0).r;
    
    if depth >= 1 {
        textureStore(outputTexture, coords, vec4f());
        return;
    }
    
    let sss_contribution = textureLoad(blurredDiffuseTexture, coords, 0).rgb;
    let base_lighting = textureLoad(baseLightingTexture, coords, 0).rgb;
    
    var final_color = base_lighting + sss_contribution * SSS_STRENGTH;
    final_color = pow(final_color, vec3f(1 / 2.2));    
    textureStore(outputTexture, coords, vec4f(final_color, 1));
}

