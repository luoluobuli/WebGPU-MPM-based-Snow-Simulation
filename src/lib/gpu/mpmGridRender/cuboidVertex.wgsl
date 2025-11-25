@vertex
fn vert(
    @location(0) pointIndex: u32,
) -> CuboidVertexOut {
    var out: CuboidVertexOut;

    const cuboidPoints: array<vec3f, 8> = array(
        vec3f(0, 0, 0),
        vec3f(1, 0, 0),
        vec3f(1, 1, 0),
        vec3f(0, 1, 0),

        vec3f(0, 0, 1),
        vec3f(1, 0, 1),
        vec3f(1, 1, 1),
        vec3f(0, 1, 1),
    );

    let pos = vec4f(mix(uniforms.gridMinCoords, uniforms.gridMaxCoords, cuboidPoints[pointIndex]), 1);

    let frustumPos = uniforms.viewProjMat * pos;

    out.posBuiltin = frustumPos;

    out.pos = pos;
    out.uv = frustumPos.xy / frustumPos.w;

    return out;
}