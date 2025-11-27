@vertex
fn vert(
    @location(0) position: vec2f,
) -> VertexOut {
    var out: VertexOut;

    out.position = vec4f(position, 0, 1);

    out.uv = position * 0.5 + 0.5;
    out.uv.y = 1 - out.uv.y;
    
    return out;
}