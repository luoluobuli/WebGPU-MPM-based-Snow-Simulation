@group(0) @binding(0) var inputTexture: texture_2d<f32>;

@fragment
fn frag(in: VertexOut) -> @location(0) vec4f {
    let dims = textureDimensions(inputTexture);
    let coords = vec2u(in.uv * vec2f(dims));
    return textureLoad(inputTexture, coords, 0);
}
