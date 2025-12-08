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
    
    
    lightViewProjMat: mat4x4f, // 368
    colliderTransformInv: mat4x4f, // 432

    // New Interaction Fields (Start 496)
    interactionPos: vec3f, // 496
    interactionStrength: f32, // 508
    interactionRadius: f32, // 512
    isInteracting: u32, // 516
    
    interactionMode: u32, // 520
    _pad_interaction2: f32, // 524 -> 528
    
    interactionDir: vec3f, // 528
    _pad_interaction3: f32, // 540 -> 544
    
    objects: array<ColliderObject, 1024>, // 544
}
