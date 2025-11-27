struct FSIn {
    @location(0) normal: vec3<f32>,
};

@fragment
fn frag(in: FSIn) -> @location(0) vec4<f32> {
    // let color = 0.5 * (normalize(in.normal) + vec3f(1.0, 1.0, 1.0));
    // return vec4f(color, 1.0);

    // directional light for a temp viewport shading
    let lightDir = normalize(vec3f(1.0, -1.0, -1.0));
    let lightColor = vec3f(1.0, 1.0, 1.0);

    let normal = normalize(in.normal);
    let diffuse = max(dot(normal, -lightDir), 0.0);
    let lighting = lightColor * diffuse;
    var color = clamp(lighting, vec3f(0.0), vec3f(1.0));
    color = mix(vec3f(0.1), vec3f(0.9), color);

    return vec4f(color, 1.0);
}
