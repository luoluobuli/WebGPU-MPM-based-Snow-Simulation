struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vert(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    const TRIANGLE_STRIP_POSITIONS = array<vec2f, 4>(
        vec2f(-1, -1),
        vec2f(-1, 1),
        vec2f(1, -1),
        vec2f(1, 1),
    );
    
    var output: VertexOutput;
    output.position = vec4f(TRIANGLE_STRIP_POSITIONS[vertexIndex], 0.0, 1.0);
    output.uv = TRIANGLE_STRIP_POSITIONS[vertexIndex] * 0.5 + 0.5;
    output.uv.y = 1.0 - output.uv.y;
    return output;
}
