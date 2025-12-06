struct ParticleData {
    // 0
    pos: vec3f, // 12
    _hom: f32, // 16; vertex shader expects a vec4
    vel: vec3f, // 28
    mass: f32, // 32
    deformationElastic: mat3x3f, // 80
    deformationPlastic: mat3x3f, // 128
    pos_displacement: vec3f, // 140
    // 144
    deformation_displacement: mat3x3f, // 192
}

fn calculateCellNumber(pos: vec3f) -> vec3i {
    let posFromGridMin = pos - uniforms.gridMinCoords;

    return vec3i(posFromGridMin / uniforms.gridCellDims);
}

fn cellNumberInGridRange(cellNumber: vec3i) -> bool {
    return all(vec3i(0) <= cellNumber) && all(cellNumber < vec3i(uniforms.gridResolution));
}

fn linearizeCellIndex(cellNumber: vec3u) -> u32 {
    return cellNumber.x + uniforms.gridResolution.x * (cellNumber.y + uniforms.gridResolution.y * cellNumber.z);
}

fn calculateFractionalPosFromCellMin(pos: vec3f, cellNumber: vec3i) -> vec3f {
    let minPos = uniforms.gridMinCoords + uniforms.gridCellDims * vec3f(cellNumber);
    return (pos - minPos) / uniforms.gridCellDims;
}


fn calculateQuadraticBSplineCellWeights(fractionalPosFromCellMin: vec3f) -> array<vec3f, 3> {
    var weights: array<vec3f, 3>;

    weights[0] = 0.5 * (1 - fractionalPosFromCellMin) * (1 - fractionalPosFromCellMin);
    weights[1] = 0.75 - (fractionalPosFromCellMin - 0.5) * (fractionalPosFromCellMin - 0.5);
    weights[2] = 0.5 * fractionalPosFromCellMin * fractionalPosFromCellMin;

    return weights;
}

fn calculateQuadraticBSplineCellWeightDerivatives(fractionalPosFromCellMin: vec3f) -> array<vec3f, 3> {
    var derivatives: array<vec3f, 3>;

    // derivative of B-spline weights wrt fractional pos
    derivatives[0] = fractionalPosFromCellMin - 1;
    derivatives[1] = -2 * (fractionalPosFromCellMin - 0.5);
    derivatives[2] = fractionalPosFromCellMin;

    return derivatives;
}
