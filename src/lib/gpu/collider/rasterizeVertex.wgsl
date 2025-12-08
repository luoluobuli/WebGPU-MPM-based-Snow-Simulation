@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VSIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) materialIndex: u32,
};

struct VSOut {
    @builtin(position) position : vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) materialIndex: u32,
    @location(3) pos: vec3f,
};

@vertex
fn vert(in: VSIn) -> VSOut {
    var out: VSOut;

    out.position = uniforms.viewProjMat * uniforms.colliderTransformMat * vec4f(in.position, 1);
    out.normal = (uniforms.colliderTransformMat * vec4<f32>(in.normal, 0.0)).xyz;
    out.uv = in.uv;
    out.materialIndex = in.materialIndex;
    out.pos = in.position;

    return out;
}