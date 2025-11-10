@fragment
fn frag(
    in: VertexOut,
) -> @location(0) vec4f {
    return vec4f(in.pos.xyz, 1);
}