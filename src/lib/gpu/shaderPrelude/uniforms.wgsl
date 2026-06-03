struct Uniforms {
    // 0

    simulationTimestep: f32, // 4
    fixedPointScale: f32, // 8
    invSimulationTimestep: f32, // 12
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
    colliderTransformIsIdentity: u32, // 356
    gridCellDims: vec3f, // 364
    colliderVelocityIsZero: u32, // 368
    
    
    lightViewProjMat: mat4x4f, // 368
    colliderTransformInv: mat4x4f, // 432

    // New Interaction Fields (Start 496)
    interactionPos: vec3f, // 496
    interactionStrength: f32, // 508
    interactionRadius: f32, // 512
    isInteracting: u32, // 516
    
    interactionMode: u32, // 520
    colliderFriction: f32, // 524 -> 528
    
    interactionDir: vec3f, // 528
    interactionRadiusSquared: f32, // 540
    maxStableParticleSpeed: f32, // 544
    maxStableParticleSpeedSquared: f32, // 548
    maxStableParticleDisplacement: f32, // 552
    maxStableParticleDisplacementSquared: f32, // 556
    colliderSdfGridScale: vec3f, // 560
    colliderSdfCellSize: vec3f, // 576
    colliderSdfValid: u32, // 588
    colliderWorldMinCoords: vec3f, // 604
    colliderWorldMaxCoords: vec3f, // 620
    colliderSdfMaxCellSize: f32, // 624
    invGridCellDims: vec3f, // 636
    invGridCellDimsSquared: vec3f, // 652
    simulationDomainCenter: vec3f, // 668
    simulationDomainMaxInside: vec3f, // 684
    gravityDeltaVelocity: vec3f, // 700
    interactionStrengthDelta: f32, // 704
    mlsDeformationGradientScale: vec3f, // 716
    explicitDeformationGradientScale: f32, // 720
    mlsStressAffineScale: vec3f, // 732
    explicitStressImpulseScale: f32, // 736
}
