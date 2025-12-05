struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

const POSITIONS: array<vec2f, 4> = array(
    vec2f(-1, -1),
    vec2f(1, -1),
    vec2f(-1, 1),
    vec2f(1, 1),
);

@vertex
fn vert(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    let pos = POSITIONS[vertex_index];
    out.position = vec4f(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + 0.5;
    
    return out;
}
