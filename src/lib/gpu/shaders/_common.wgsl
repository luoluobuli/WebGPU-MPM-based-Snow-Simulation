struct VertexOut {
    @builtin(position) posBuiltin: vec4f,
    @location(0) pos: vec4f,
    @location(1) uv: vec2f,
}

struct Uniforms {
    // 0

    simulationTimestep: f32, // 4
    gridResolution: u32, // 8
    fixedPointScale: f32, // 12
    // 16
    gridMinCoords: vec3f, // 28
    // 32
    gridMaxCoords: vec3f, // 44
    // 48
    viewInvProjMat: mat4x4f, // 112
}

struct ParticleData {
    // 0
    pos: vec3f, // 12
    _hom: f32, // 16; vertex shader expects a vec4
    vel: vec3f, // 28
    // 32
    affine: vec3f, // 44
    mass: f32, // 48
}

struct GridData {
    // 0

    // vel: vec3f, // 12
    // mass: f32, // 16
    vx: atomic<i32>, // 4
    vy: atomic<i32>, // 8
    vz: atomic<i32>, // 12
    mass: atomic<i32>, // 16
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
fn cellNumberThatContainsPos(pos: vec3f) -> vec3i {
    let gridCellSize = (uniforms.gridMaxCoords - uniforms.gridMinCoords) / f32(uniforms.gridResolution);
    let posFromGridOrigin = pos - uniforms.gridMinCoords;
    return vec3i(
        i32(posFromGridOrigin.x / gridCellSize.x),
        i32(posFromGridOrigin.y / gridCellSize.y),
        i32(posFromGridOrigin.z / gridCellSize.z),
    );
}