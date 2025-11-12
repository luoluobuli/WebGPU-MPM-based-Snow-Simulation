@fragment
fn frag(
    in: VertexOut,
) -> @location(0) vec4f {
    return vec4f(in.pos.xyz * 0.5 + 0.5, 1);
}