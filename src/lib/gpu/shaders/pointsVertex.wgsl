@vertex
fn vert(
    @location(0) pos: vec4f,
) -> PointsVertexOut {
    var out: PointsVertexOut;

    let frustumPos: vec4f = uniforms.viewInvProjMat * pos;

    out.posBuiltin = frustumPos;

    out.pos = pos;
    out.uv = frustumPos.xy / frustumPos.w;

    return out;
}