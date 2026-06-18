@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(0) @binding(1) var<storage, read> particle_data: array<ParticleData>;
@group(0) @binding(2) var<storage, read> particle_appearance: array<u32>;
@group(0) @binding(3) var<storage, read> particle_flags: array<u32>;

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
    let particle = particle_data[vertex_index];
    let flags = particle_flags[vertex_index];
    out.deformation_elastic_volume = particleElasticVolumeFactor(particle, flags);
    out.deformation_plastic_volume = particlePlasticVolumeFactor(particle, flags);
    out.appearance = decodeParticleAppearance(particle_appearance[vertex_index]);

    return out;
}
