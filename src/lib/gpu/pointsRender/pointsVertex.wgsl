@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(0) @binding(1) var<storage, read> particle_data: array<ParticleData>;

@vertex
fn vert(
    @location(0) pos: vec4f,
    @builtin(vertex_index) vertex_index: u32,
) -> PointsVertexOut {
    var out: PointsVertexOut;

    let frustumPos: vec4f = uniforms.viewProjMat * pos;

    out.posBuiltin = frustumPos;

    out.pos = pos;
    out.uv = frustumPos.xy / frustumPos.w;
    out.deformation_elastic_volume = determinant(particle_data[vertex_index].deformationElastic);
    out.deformation_plastic_volume = determinant(particle_data[vertex_index].deformationPlastic);

    return out;
}