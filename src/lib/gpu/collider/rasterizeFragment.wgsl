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
    // Sample texture array using UV and material index
    let texColor = textureSample(texArray, texSampler, in.uv, in.materialIndex);
    
    // Directional light for diffuse shading
    let lightDir = normalize(vec3f(1.0, -1.0, -1.0));
    let lightColor = vec3f(1.0, 1.0, 1.0);

    let normal = normalize(in.normal);
    let diffuse = max(dot(normal, -lightDir), 0.0);
    
    // Ambient + diffuse lighting
    let ambient = 0.2;
    let lighting = ambient + (1.0 - ambient) * diffuse;
    
    // Apply texture color as diffuse material
    var color = texColor.rgb * lightColor * lighting;
    color = clamp(color, vec3f(0.0), vec3f(1.0));

    return vec4f(color, texColor.a);
}
