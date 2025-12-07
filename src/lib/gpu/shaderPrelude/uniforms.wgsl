struct ColliderObject {
    min: vec3f,
    startIndex: u32,
    max: vec3f,
    countIndices: u32,
}

struct Uniforms {
    // 0

    simulationTimestep: f32, // 4
    fixedPointScale: f32, // 8
    use_pbmpm: u32, // 12
    time: u32, // 16
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
    colliderMinCoords: vec3f, // 236
    // 240
    colliderMaxCoords: vec3f, // 252
    // 256
    colliderTransformMat: mat4x4f, // 320
    colliderVelocity: vec3f, // 336
    // 340
    cameraPos: vec3f, // 352
    colliderNumIndices: u32, // 356
    gridCellDims: vec3f, // 364
    colliderNumObjects: u32, // 368
    // 380 (padding to 384)
    lightViewProjMat: mat4x4f, // 448
    colliderTransformInv: mat4x4f, // 512

    dynamicColliderMinCoords: vec3f, // 528
    dynamicColliderMaxCoords: vec3f, // 540
    dynamicColliderNumObjects: u32, // 544

    objects: array<ColliderObject, 1024>, // 33312
    dynamicObjects: array<ColliderObject, 512> // 49696
}
