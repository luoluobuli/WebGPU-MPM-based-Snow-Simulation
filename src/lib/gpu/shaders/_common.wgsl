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


@group(0) @binding(0) var<uniform> uniforms: Uniforms;


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



fn mat3x3Inverse(matrix: mat3x3f) -> mat3x3f {
    let det = determinant(matrix);
    if det == 0 {
        return mat3x3f(
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        );
    }
    
    return mat3x3f(
        matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1],
        matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2],
        matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1],
        
        matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2],
        matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0],
        matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2],
        
        matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0],
        matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1],
        matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
    ) * (1 / det);
}


// iterative polar decomposition approximates pure rotation R from F where F = RS
fn calculatePolarDecompositionRotation(deformation: mat3x3f) -> mat3x3f {
    var rotationGuess = deformation;
    
    const N_POLAR_DECOMPOSITION_ITERATIONS = 4;
    for (var i = 0u; i < N_POLAR_DECOMPOSITION_ITERATIONS; i++) {
        let rotationGuessInverseTranspose = transpose(mat3x3Inverse(rotationGuess));
        rotationGuess = 0.5 * (rotationGuess + rotationGuessInverseTranspose);
    }
    
    return rotationGuess;
}

// first Piola-Kirchhoff stress tensor
fn calculateStressFirstPiolaKirchhoff( // P
    deformation: mat3x3f, // F
    shearModulus: f32, // μ
    bulkModulus: f32, // λ
) -> mat3x3f {
    let volumeScaleFac = determinant(deformation); // J

    // corotation to separate out rotation from scaling
    let rotation = calculatePolarDecompositionRotation(deformation); // R
    
    // P = 2 μ (F - R) + λ (J - 1) J (F⁻¹)ᵀ
    return 2 * shearModulus * (deformation - rotation)
        + bulkModulus * (volumeScaleFac - 1) * volumeScaleFac * transpose(mat3x3Inverse(deformation));
}

// Neo-Hookean constitutive model
fn calculateStressNeoHookean(
    deformation: mat3x3f, // F
    shearModulus: f32, // μ
    bulkModulus: f32, // λ
) -> mat3x3f {
    let volumeScaleFac = determinant(deformation); // J
    let deformationInverseTranspose = transpose(mat3x3Inverse(deformation));
    
    // P = μ (F - (F⁻¹)ᵀ) + λ log(J) (F⁻¹)ᵀ
    return shearModulus * (deformation - deformationInverseTranspose) + bulkModulus * log(volumeScaleFac) * deformationInverseTranspose;
}