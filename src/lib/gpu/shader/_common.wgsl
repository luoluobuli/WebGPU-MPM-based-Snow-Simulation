struct VertexOut {
    @builtin(position) posBuiltin: vec4f,
    @location(0) pos: vec4f,
    @location(1) uv: vec2f,
}

struct Uniforms {
    // 0

    viewInvProjMat: mat4x4f, // 64
}

struct ParticleData {
    // 0

    pos: vec3f, // 12
    _dummy: f32, // 16
    vel: vec3f, // 28
    mass: f32, // 32
}