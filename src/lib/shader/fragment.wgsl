@fragment
fn frag(
    data: VertexOut,
) -> @location(0) vec4f {
    return vec4f(data.uv, 0, 1);
}