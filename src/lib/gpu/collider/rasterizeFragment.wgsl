@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var texSampler: sampler;
@group(1) @binding(1) var texArray: texture_2d_array<f32>;

struct FSIn {
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) materialIndex: u32,
    @location(3) pos: vec3f,
};

@fragment
fn frag(in: FSIn) -> @location(0) vec4<f32> {
    let texutre_color = textureSample(texArray, texSampler, in.uv, in.materialIndex);
    let texture_color_linear = pow(texutre_color.rgb, vec3f(2.2));
    
    // Directional light for diffuse shading
    let lightDir = normalize(vec3f(1.0, -1.0, -1.0));
    let lightColor = vec3f(0.97, 0.99, 1);

    var normal = normalize(in.normal);
    normal = normalize(normal - fbmNoise(in.pos * 16, 8) * 0.2);

    let diffuse = max(-dot(normal, lightDir), 0) * texture_color_linear;
    
    const AMBIENT_COLOR = vec3f(0.1, 0.12, 0.15);
    let lighting = AMBIENT_COLOR + diffuse;
    
    var color = lightColor * lighting;
    color = clamp(color, vec3f(0.0), vec3f(1.0));

    return vec4f(pow(color, vec3f(1 / 2.2)), texutre_color.a);
}
