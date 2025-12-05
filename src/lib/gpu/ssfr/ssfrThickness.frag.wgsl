struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) compression_volume_fac: f32,
}

const THICKNESS_CONTRIBUTION = 0.001;

@fragment
fn frag(in: VertexOutput) -> @location(0) vec4f {
    let density_factor = mix(1, 0.3, saturate(1 - in.compression_volume_fac));
    let thickness = THICKNESS_CONTRIBUTION * density_factor;
    return vec4f(thickness, thickness, thickness, thickness);
}
