// Marching cubes mesh vertex shader

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}

struct VertexOutput {
    @builtin(position) clipPos: vec4f,
    @location(0) worldPos: vec3f,
    @location(1) normal: vec3f,
}


@vertex
fn vert(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clipPos = uniforms.viewProjMat * vec4f(input.position, 1.0);
    output.worldPos = input.position;
    output.normal = input.normal;
    return output;
}
