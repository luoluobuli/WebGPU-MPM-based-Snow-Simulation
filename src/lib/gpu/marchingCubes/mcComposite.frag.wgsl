// Composite fragment shader

struct FragmentInput {
    @location(0) uv: vec2f,
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

@group(0) @binding(0) var shadedTexture: texture_2d<f32>;
@group(0) @binding(1) var depthTexture: texture_depth_2d;

@fragment
fn frag(input: FragmentInput) -> FragmentOutput {
    let coords = vec2i(input.uv * vec2f(textureDimensions(shadedTexture)));
    let shaded = textureLoad(shadedTexture, coords, 0);
    let depth = textureLoad(depthTexture, coords, 0);
    
    var output: FragmentOutput;
    
    if shaded.a < 0.01 {
        discard;
    }
    
    output.color = shaded;
    output.depth = depth;
    return output;
}
