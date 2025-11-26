struct FSIn {
    @location(0) normal: vec3<f32>,
};

@fragment
fn frag(in: FSIn) -> @location(0) vec4<f32> {
    let color = 0.5 * (normalize(in.normal) + vec3f(1.0, 1.0, 1.0));
    return vec4f(color, 1.0);
}
