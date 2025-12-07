@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@fragment
fn frag(
    in: PointsVertexOut,
) -> @location(0) vec4f {
    let blend_elastic_fac = log(in.deformation_elastic_volume) * 240 + 0.85;
    let blend_plastic_fac = log(in.deformation_plastic_volume) * 24 + 0.85;

    return vec4f(
        mix(0, 1, blend_elastic_fac),
        mix(0, 1, blend_plastic_fac),
        0.85,
        1,
    );
}