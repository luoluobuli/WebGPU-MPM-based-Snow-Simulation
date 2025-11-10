@vertex
fn vert(
    @location(0) pos: vec4f,
) -> VertexOut {
    var out: VertexOut;

    out.pos = pos;
    out.uv = pos.xy / pos.z;

    return out;
}