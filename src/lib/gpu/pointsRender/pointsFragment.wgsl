@fragment
fn frag(
    in: PointsVertexOut,
) -> @location(0) vec4f {
    let blend_elastic_fac = log(in.deformation_elastic_volume) * 20 + 0.5;
    let blend_plastic_fac = log(in.deformation_plastic_volume) * 2 + 0.5;

    return vec4f(
        mix(0, 1, blend_elastic_fac),
        mix(0, 1, blend_plastic_fac),
        0.5,
        1,
    );
}