@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VSIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VSOut {
    @builtin(position) position : vec4<f32>,
    @location(0) normal: vec3<f32>,
};

@vertex
fn vert(in: VSIn) -> VSOut {
    var out: VSOut;
    out.position = uniforms.viewProjMat * uniforms.colliderTransformMat * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    return out;
}