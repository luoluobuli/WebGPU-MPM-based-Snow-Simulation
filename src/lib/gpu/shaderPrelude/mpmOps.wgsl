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

fn calculateGridCoordinate(pos: vec3f) -> vec3f {
    return (pos - uniforms.gridMinCoords) / uniforms.gridCellDims;
}

fn calculateCellNumber(pos: vec3f) -> vec3i {
    return vec3i(floor(calculateGridCoordinate(pos)));
}

fn isFiniteScalar(value: f32) -> bool {
    return value == value && abs(value) < 1e20;
}

fn isFiniteVec3(value: vec3f) -> bool {
    return isFiniteScalar(value.x) && isFiniteScalar(value.y) && isFiniteScalar(value.z);
}

fn isFiniteMat3(value: mat3x3f) -> bool {
    return isFiniteVec3(value[0]) && isFiniteVec3(value[1]) && isFiniteVec3(value[2]);
}

fn sanitizeScalar(value: f32, fallback: f32) -> f32 {
    if isFiniteScalar(value) {
        return value;
    }

    return fallback;
}

fn sanitizeVec3(value: vec3f, fallback: vec3f) -> vec3f {
    return vec3f(
        sanitizeScalar(value.x, fallback.x),
        sanitizeScalar(value.y, fallback.y),
        sanitizeScalar(value.z, fallback.z),
    );
}

fn maxStableParticleSpeed() -> f32 {
    let min_cell_dim = min(uniforms.gridCellDims.x, min(uniforms.gridCellDims.y, uniforms.gridCellDims.z));
    return 2.0 * min_cell_dim / max(uniforms.simulationTimestep, 1e-6);
}

fn clampVec3Length(value: vec3f, maxLength: f32) -> vec3f {
    let safe_value = sanitizeVec3(value, vec3f(0.0));
    let len = length(safe_value);
    if len > maxLength {
        return safe_value * (maxLength / len);
    }

    return safe_value;
}

fn clampMatrixComponents(value: mat3x3f, limit: f32) -> mat3x3f {
    let lower = vec3f(-limit);
    let upper = vec3f(limit);
    return mat3x3f(
        clamp(value[0], lower, upper),
        clamp(value[1], lower, upper),
        clamp(value[2], lower, upper),
    );
}

fn sanitizeDeformationDelta(value: mat3x3f) -> mat3x3f {
    if !isFiniteMat3(value) {
        return mat3x3f();
    }

    return clampMatrixComponents(value, 0.35);
}

fn simulationDomainMaxInside() -> vec3f {
    return uniforms.gridMaxCoords - uniforms.gridCellDims * 0.001;
}

fn clampPositionToSimulationDomain(pos: vec3f) -> vec3f {
    let domain_center = 0.5 * (uniforms.gridMinCoords + uniforms.gridMaxCoords);
    return clamp(sanitizeVec3(pos, domain_center), uniforms.gridMinCoords, simulationDomainMaxInside());
}

fn particlePositionCanTouchGrid(pos: vec3f) -> bool {
    return isFiniteVec3(pos)
        && all(uniforms.gridMinCoords <= pos)
        && all(pos < uniforms.gridMaxCoords);
}

fn sanitizeParticle(particle: ptr<function, ParticleData>) {
    (*particle)._hom = 1.0;
    (*particle).pos = clampPositionToSimulationDomain((*particle).pos);

    if !isFiniteScalar((*particle).mass) || (*particle).mass <= 0.0 {
        (*particle).mass = 1.0 / 3.0;
    }

    let max_speed = maxStableParticleSpeed();
    (*particle).vel = clampVec3Length((*particle).vel, max_speed);
    (*particle).pos_displacement = clampVec3Length((*particle).pos_displacement, max_speed * uniforms.simulationTimestep);
    (*particle).deformation_displacement = sanitizeDeformationDelta((*particle).deformation_displacement);

    let elastic_det = determinant((*particle).deformationElastic);
    let plastic_det = determinant((*particle).deformationPlastic);
    if !isFiniteMat3((*particle).deformationElastic)
        || elastic_det != elastic_det
        || elastic_det < 0.05
        || elastic_det > 20.0
    {
        (*particle).deformationElastic = IDENTITY_MAT3;
        (*particle).deformation_displacement = mat3x3f();
    }

    if !isFiniteMat3((*particle).deformationPlastic)
        || plastic_det != plastic_det
        || plastic_det < 0.05
        || plastic_det > 20.0
    {
        (*particle).deformationPlastic = IDENTITY_MAT3;
    }
}

fn toFixedPointI32(value: f32) -> i32 {
    let scaled_value = sanitizeScalar(value, 0.0) * uniforms.fixedPointScale;
    return i32(clamp(scaled_value, -2147483000.0, 2147483000.0));
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

fn calculateCellGridOffsetFromParticle(cellNumber: vec3i, particlePos: vec3f) -> vec3f {
    return vec3f(cellNumber) - calculateGridCoordinate(particlePos) + vec3f(0.5);
}

fn calculateCellWorldOffsetFromParticle(cellNumber: vec3i, particlePos: vec3f) -> vec3f {
    return calculateCellGridOffsetFromParticle(cellNumber, particlePos) * uniforms.gridCellDims;
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
