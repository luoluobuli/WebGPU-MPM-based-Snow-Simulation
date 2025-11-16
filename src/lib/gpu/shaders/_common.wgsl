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
    viewProjMat: mat4x4f, // 112
    viewProjInvMat: mat4x4f, // 176
    // 180
    meshMinCoords: vec3f, // 192
    // 196
    meshMaxCoords: vec3f, // 208
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


fn calculateCellDims() -> vec3f {
    return (uniforms.gridMaxCoords - uniforms.gridMinCoords) / f32(uniforms.gridResolution);
}


fn calculateMpmParticleCellInfo(pos: vec3f) -> MpmParticleCellInfo {
    let cellDims = calculateCellDims();
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

// https://github.com/Cyan4973/xxHash
// https://gist.github.com/munrocket/236ed5ba7e409b8bdf1ff6eca5dcdc39
fn hash1(n: u32) -> u32 {
    var h32 = n + 374761393u;
    h32 = 668265263u * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = 2246822519u * (h32 ^ (h32 >> 15));
    h32 = 3266489917u * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

fn hash3(p: vec3u) -> u32 {
    let p2 = 2246822519u;
    let p3 = 3266489917u;
    let p4 = 668265263u;
    let p5 = 374761393u;

    var h32 = p.z + p5 + p.x*p3;
    h32 = p4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * p3;
    h32 = p4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = p2 * (h32 ^ (h32 >> 15));
    h32 = p3 * (h32 ^ (h32 >> 13));
    
    return h32 ^ (h32 >> 16);
}

fn hash4(p: vec4u) -> u32 {
    let p2 = 2246822519u;
    let p3 = 3266489917u;
    let p4 = 668265263u;
    let p5 = 374761393u;

    var h32 = p.w + p5 + p.x * p3;
    h32 = p4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * p3;
    h32 = p4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.z  * p3;
    h32 = p4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = p2 * (h32 ^ (h32 >> 15));
    h32 = p3 * (h32 ^ (h32 >> 13));

    return h32 ^ (h32 >> 16);
}