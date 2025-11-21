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


// iterative polar decomposition approximates pure rotation R from F where F = R S
fn calculatePolarDecompositionRotation(deformation: mat3x3f) -> mat3x3f {
    var rotationGuess = deformation; // R_n
    
    const N_ITERATIONS = 8;
    for (var i = 0u; i < N_ITERATIONS; i++) {
        // idea: the closer R_n gets to the true rotation R, the closer the inverse transpose should be to R as well
        let rotationGuessInverseTranspose = transpose(mat3x3Inverse(rotationGuess));
        // so average toward it
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
fn calculateStressNeoHookean( // P
    deformation: mat3x3f, // F
    shearModulus: f32, // μ
    bulkModulus: f32, // λ
) -> mat3x3f {
    let volumeScaleFac = determinant(deformation); // J
    let deformationInverseTranspose = transpose(mat3x3Inverse(deformation));
    
    // P = μ (F - (F⁻¹)ᵀ) + λ log(J) (F⁻¹)ᵀ
    return shearModulus * (deformation - deformationInverseTranspose) + bulkModulus * log(volumeScaleFac) * deformationInverseTranspose;
}

/**
 * Checks whether a particle's elastic deformation indicates that the particle should undergo permanent (plastic)
 * deformation. This check is done by comparing the amount of stretching (non-uniform scaling) to a minimum stretch
 * factor and maximum stretch factor. If an actual stretch factor in some direction lies outside of this range, the
 * deformation matrices are updated to use a clamped amount of stretching in that direction.
 */
fn applyPlasticity(
    deformationElastic: ptr<function, mat3x3f>, // F_e
    deformationPlastic: ptr<function, mat3x3f>, // F_p
) {
    // extract rotation and scale from the elastic deformation
    let rotation = calculatePolarDecompositionRotation(*deformationElastic); // R
    // S = Rᵀ F_e  (<=  F_e = R S) 
    let scale = transpose(rotation) * (*deformationElastic); // S
    
    // thresholds of compression/stretching until yielding
    const MIN_STRETCH_FAC = 1 - 2.5e-2; // θ_c
    const MAX_STRETCH_FAC = 1 + 7.5e-3; // θ_s

    // singulars = individual scale factors extracted from the scale matrix (approximate)
    let singulars = vec3f(scale[0][0], scale[1][1], scale[2][2]); // σ_0, σ_1, σ_2


    // compare the singulars to the range to know whether the material yields
    if all(vec3f(MIN_STRETCH_FAC) <= singulars) && all(singulars <= vec3f(MAX_STRETCH_FAC)) { return; }


    // clamped scale factors represent how much of the compression or stretching can actually be restored later
    let singularsClamped = clamp(singulars, vec3f(MIN_STRETCH_FAC), vec3f(MAX_STRETCH_FAC));

    // construct a new scale matrix from the clamped values
    let newScale = mat3x3f(
        singularsClamped.x, 0, 0,
        0, singularsClamped.y, 0,
        0, 0, singularsClamped.z,
    );
    // assuming the rotation stays the same, we can reconstruct the new elastic deformation
    let newDeformationElastic = rotation * newScale; // F_e
    
    // assuming the overall deformaton stays the same, we can also derive the new plastic deformation
    let newDeformationElasticInv = mat3x3Inverse(newDeformationElastic);
    let deformation = (*deformationElastic) * (*deformationPlastic);
    
    *deformationElastic = newDeformationElastic;
    // F_p = (F_e)⁻¹ F  (<=  F = F_e F_p)  
    *deformationPlastic = newDeformationElasticInv * deformation;
}