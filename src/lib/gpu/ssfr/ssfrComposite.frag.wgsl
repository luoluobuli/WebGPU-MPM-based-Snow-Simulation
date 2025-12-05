struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@group(0) @binding(1) var shadedTexture: texture_2d<f32>;
@group(0) @binding(2) var smoothedDepthTexture: texture_2d<f32>;

@fragment
fn frag(in: VertexOutput) -> @location(0) vec4f {
    let dims = textureDimensions(shadedTexture);

    let uv = vec2f(in.uv.x, 1 - in.uv.y);

    let coords = vec2u(uv * vec2f(dims));
    
    let color = textureLoad(shadedTexture, coords, 0);
    let depth = textureLoad(smoothedDepthTexture, vec2i(coords), 0).r;
    
    if color.a < 0.01 || depth >= 1.0 {discard; }
    
    return color;
}
