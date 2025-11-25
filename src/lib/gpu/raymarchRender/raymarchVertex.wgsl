@vertex
fn vert(
    @location(0) uvCentered: vec2f,
) -> RaymarchVertexOut {
    var out: RaymarchVertexOut;

    out.posBuiltin = vec4f(uvCentered, 0, 1);
    out.uvCentered = uvCentered;

    return out;
}