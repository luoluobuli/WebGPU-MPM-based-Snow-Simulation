@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vert(
    @location(0) pos: vec4f,
) -> VertexOut {
    var out: VertexOut;

    let frustumPos: vec4f = uniforms.viewInvProjMat * pos;

    out.posBuiltin = frustumPos;

    out.pos = pos;
    out.uv = frustumPos.xy / frustumPos.w;

    return out;
}