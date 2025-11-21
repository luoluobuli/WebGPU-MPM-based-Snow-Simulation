struct ParticleData {
    // 0
    pos: vec3f, // 12
    _hom: f32, // 16; vertex shader expects a vec4
    vel: vec3f, // 28
    mass: f32, // 32
    deformationElastic: mat3x3f, // 80
    deformationPlastic: mat3x3f, // 128
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




fn calculateCellDims() -> vec3f {
    return (uniforms.gridMaxCoords - uniforms.gridMinCoords) / f32(uniforms.gridResolution);
}

fn calculateCellNumber(pos: vec3f, cellDims: vec3f) -> vec3i {
    let posFromGridMin = pos - uniforms.gridMinCoords;

    return vec3i(
        i32(posFromGridMin.x / cellDims.x),
        i32(posFromGridMin.y / cellDims.y),
        i32(posFromGridMin.z / cellDims.z),
    );
}

fn cellNumberInGridRange(cellNumber: vec3i) -> bool {
    return all(vec3i(0) <= cellNumber) && all(cellNumber < vec3i(i32(uniforms.gridResolution)));
}

fn linearizeCellIndex(cellNumber: vec3i) -> u32 {
    return u32(cellNumber.x) + uniforms.gridResolution * (u32(cellNumber.y) + uniforms.gridResolution * u32(cellNumber.z));
}

fn calculateFractionalPosFromCellMin(pos: vec3f, cellDims: vec3f, cellNumber: vec3i) -> vec3f {
    let minPos = uniforms.gridMinCoords + cellDims * vec3f(cellNumber);
    return (pos - minPos) / cellDims;
}


fn calculateQuadraticBSplineCellWeights(fractionalPosFromCellMin: vec3f) -> array<vec3f, 3> {
    var weights: array<vec3f, 3>;

    weights[0] = 0.5 * (0.5 - fractionalPosFromCellMin) * (0.5 - fractionalPosFromCellMin);
    weights[1] = 0.75 - fractionalPosFromCellMin * fractionalPosFromCellMin;
    weights[2] = 0.5 * (0.5 + fractionalPosFromCellMin) * (0.5 + fractionalPosFromCellMin);

    return weights;
}

fn calculateQuadraticBSplineCellWeightDerivatives(fractionalPosFromCellMin: vec3f) -> array<vec3f, 3> {
    var derivatives: array<vec3f, 3>;

    // derivative of B-spline weights wrt fractional pos
    derivatives[0] = fractionalPosFromCellMin - 0.5;
    derivatives[1] = -2 * fractionalPosFromCellMin;
    derivatives[2] = fractionalPosFromCellMin + 0.5;

    return derivatives;
}
