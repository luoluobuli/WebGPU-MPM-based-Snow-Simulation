struct Uniforms {
    // 0

    simulationTimestep: f32, // 4
    fixedPointScale: f32, // 8
    // 16
    gridMinCoords: vec3f, // 28
    // 32
    gridMaxCoords: vec3f, // 44
    // 48
    viewProjMat: mat4x4f, // 112
    viewProjInvMat: mat4x4f, // 176
    meshMinCoords: vec3f, // 188
    // 192
    meshMaxCoords: vec3f, // 204
    // 208
    gridResolution: vec3u, // 220
    // 224
    min: vec3f, // 236
    // 240
    max: vec3f, // 252
    // 256
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
