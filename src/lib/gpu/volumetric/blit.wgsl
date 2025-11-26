struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vert(@location(0) position: vec2f) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(position, 0.0, 1.0);
    // Map [-1, 1] to [0, 1]
    out.uv = position * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y; // Flip Y if needed, or handle in compute
    return out;
}

@group(0) @binding(0) var inputTexture: texture_2d<f32>;

@fragment
fn frag(in: VertexOutput) -> @location(0) vec4f {
    // Load integer coordinates
    let dims = textureDimensions(inputTexture);
    let coords = vec2u(in.uv * vec2f(dims));
    return textureLoad(inputTexture, coords, 0);
}
