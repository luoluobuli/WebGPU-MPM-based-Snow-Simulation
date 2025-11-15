struct PointsVertexOut {
    @builtin(position) posBuiltin: vec4f,
    @location(0) pos: vec4f,
    @location(1) uv: vec2f,
}

struct RaymarchVertexOut {
    @builtin(position) posBuiltin: vec4f,
    @location(0) uvCentered: vec2f,
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
    viewInvMat: mat4x4f, // 176
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

struct CellData {
    // 0

    // vel: vec3f, // 12
    // mass: f32, // 16
    momentumX: atomic<i32>, // 4
    momentumY: atomic<i32>, // 8
    momentumZ: atomic<i32>, // 12
    mass: atomic<i32>, // 16
}


struct MpmParticleCellInfo {
    startCellNumber: vec3i,
    velocityWeightsKernel: array<vec3f, 3>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;


fn calculateMpmParticleCellInfo(pos: vec3f) -> MpmParticleCellInfo {
    let cellDims = (uniforms.gridMaxCoords - uniforms.gridMinCoords) / f32(uniforms.gridResolution);
    let posFromGridMin = pos - uniforms.gridMinCoords;

    let cellNumber = vec3i(
        i32(posFromGridMin.x / cellDims.x),
        i32(posFromGridMin.y / cellDims.y),
        i32(posFromGridMin.z / cellDims.z),
    );

    let minPos = uniforms.gridMinCoords + cellDims * vec3f(cellNumber);
    let fractionalPosFromCellMin = (pos - minPos) / cellDims;



    var particleInfo: MpmParticleCellInfo;

    particleInfo.startCellNumber = cellNumber;

    // values from quadratic B-spline weighting
    particleInfo.velocityWeightsKernel[0] = 0.5 * (1.5 - fractionalPosFromCellMin) * (1.5 - fractionalPosFromCellMin);
    particleInfo.velocityWeightsKernel[1] = 0.75 - (fractionalPosFromCellMin - 1.0) * (fractionalPosFromCellMin - 1.0);
    particleInfo.velocityWeightsKernel[2] = 0.5 * (fractionalPosFromCellMin - 0.5) * (fractionalPosFromCellMin - 0.5);

    return particleInfo;
}