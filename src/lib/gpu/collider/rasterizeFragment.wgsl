@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var texSampler: sampler;
@group(1) @binding(1) var texArray: texture_2d_array<f32>;

struct FSIn {
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) materialIndex: u32,
};

@fragment
fn frag(in: FSIn) -> @location(0) vec4<f32> {
    let texutre_color = textureSample(texArray, texSampler, in.uv, in.materialIndex);
    let texture_color_linear = pow(texutre_color.rgb, vec3f(2.2));
    
    // Directional light for diffuse shading
    let lightDir = normalize(vec3f(1.0, -1.0, -1.0));
    let lightColor = vec3f(1.0, 1.0, 1.0);

    let normal = normalize(in.normal);
    let diffuse = max(dot(normal, -lightDir), 0.0);
    
    const AMBIENT_COLOR = vec3f(0.3, 0.32, 0.35);
    let lighting = AMBIENT_COLOR + (1.0 - AMBIENT_COLOR) * diffuse;
    
    var color = texture_color_linear * lightColor * lighting;
    color = clamp(color, vec3f(0.0), vec3f(1.0));

    return vec4f(pow(color, vec3f(1 / 2.2)), texutre_color.a);
}
