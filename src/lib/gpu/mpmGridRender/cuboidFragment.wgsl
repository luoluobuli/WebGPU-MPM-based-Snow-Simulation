@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@fragment
fn frag() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}