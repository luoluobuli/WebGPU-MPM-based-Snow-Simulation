struct VertexOut {
    @builtin(position) posBuiltin: vec4f,
    @location(0) pos: vec4f,
    @location(1) uv: vec2f,
}

struct Uniforms {
    // 0

    simulationTimestep: f32, // 4
    // 16
    viewInvProjMat: mat4x4f, // 80
}

struct ParticleData {
    // 0
    pos: vec3f, // 12
    _hom: f32, // 16; vertex shader expects a vec4
    vel: vec3f, // 28
    // 32
    deform: mat3x3f, // 80
}

struct GridData {
    // 0

    vel: vec3f, // 12
    mass: f32, // 16
}