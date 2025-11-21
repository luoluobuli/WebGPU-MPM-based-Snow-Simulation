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
    viewProjMat: mat4x4f, // 112
    viewProjInvMat: mat4x4f, // 176
    // 180
    meshMinCoords: vec3f, // 192
    // 196
    meshMaxCoords: vec3f, // 208
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
