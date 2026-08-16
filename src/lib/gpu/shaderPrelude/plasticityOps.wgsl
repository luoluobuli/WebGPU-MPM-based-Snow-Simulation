struct Svd3x3 {
    left: mat3x3f,
    singulars: vec3f,
    right: mat3x3f,
}

const SVD_JACOBI_SWEEPS = 6u;
const SVD_RELATIVE_TOLERANCE = 1e-6;
const SVD_MIN_NORMALIZED_SINGULAR = 1e-7;
const DEFAULT_CRITICAL_COMPRESSION = 2.5e-2;
const DEFAULT_CRITICAL_STRETCH = 7.5e-3;
const CRUST_CRITICAL_COMPRESSION = 6.0e-2;
const CRUST_CRITICAL_STRETCH = 2.0e-2;

fn rotateSvdColumns(
    diagonalized: ptr<function, mat3x3f>,
    right: ptr<function, mat3x3f>,
    p: u32,
    q: u32,
) {
    let diagonalizedP = (*diagonalized)[p];
    let diagonalizedQ = (*diagonalized)[q];
    let alpha = dot(diagonalizedP, diagonalizedP);
    let beta = dot(diagonalizedQ, diagonalizedQ);
    let gamma = dot(diagonalizedP, diagonalizedQ);
    let tolerance = SVD_RELATIVE_TOLERANCE * sqrt(max(alpha * beta, 0.0));

    if abs(gamma) <= tolerance {
        return;
    }

    // This stable tangent avoids the cancellation in the direct atan solution.
    let tau = (beta - alpha) / (2.0 * gamma);
    let stableTangent = sign(tau) / (abs(tau) + sqrt(1.0 + tau * tau));
    let tangent = select(1.0, stableTangent, abs(tau) > SVD_MIN_NORMALIZED_SINGULAR);
    let cosine = inverseSqrt(1.0 + tangent * tangent);
    let sine = tangent * cosine;
    let rightP = (*right)[p];
    let rightQ = (*right)[q];

    (*diagonalized)[p] = cosine * diagonalizedP - sine * diagonalizedQ;
    (*diagonalized)[q] = sine * diagonalizedP + cosine * diagonalizedQ;
    (*right)[p] = cosine * rightP - sine * rightQ;
    (*right)[q] = sine * rightP + cosine * rightQ;
}

fn swapSvdColumns(
    diagonalized: ptr<function, mat3x3f>,
    right: ptr<function, mat3x3f>,
    p: u32,
    q: u32,
) {
    let diagonalizedP = (*diagonalized)[p];
    let rightP = (*right)[p];

    (*diagonalized)[p] = (*diagonalized)[q];
    (*diagonalized)[q] = diagonalizedP;
    (*right)[p] = (*right)[q];
    (*right)[q] = rightP;
}

fn perpendicularUnitVector(direction: vec3f) -> vec3f {
    let directionAbs = abs(direction);
    var leastAlignedAxis = vec3f(1.0, 0.0, 0.0);

    if directionAbs.y <= directionAbs.x && directionAbs.y <= directionAbs.z {
        leastAlignedAxis = vec3f(0.0, 1.0, 0.0);
    } else if directionAbs.z <= directionAbs.x && directionAbs.z <= directionAbs.y {
        leastAlignedAxis = vec3f(0.0, 0.0, 1.0);
    }

    return normalize(cross(direction, leastAlignedAxis));
}

/**
 * Computes F = U Sigma V^T with a one-sided Jacobi SVD.
 *
 * Scaling F before the sweeps avoids overflow in the column dot products.
 * Sorting lets the Gram-Schmidt completion handle rank-deficient inputs from
 * strongest to weakest direction.
 */
fn calculateSvd3x3(deformation: mat3x3f) -> Svd3x3 {
    let deformationScale = max(
        max(
            max(abs(deformation[0].x), abs(deformation[0].y)),
            max(abs(deformation[0].z), abs(deformation[1].x)),
        ),
        max(
            max(abs(deformation[1].y), abs(deformation[1].z)),
            max(
                max(abs(deformation[2].x), abs(deformation[2].y)),
                abs(deformation[2].z),
            ),
        ),
    );

    if !(deformationScale > 0.0) {
        return Svd3x3(IDENTITY_MAT3, vec3f(0.0), IDENTITY_MAT3);
    }

    var diagonalized = deformation * (1.0 / deformationScale);
    var right = IDENTITY_MAT3;

    for (var sweep = 0u; sweep < SVD_JACOBI_SWEEPS; sweep++) {
        rotateSvdColumns(&diagonalized, &right, 0u, 1u);
        rotateSvdColumns(&diagonalized, &right, 0u, 2u);
        rotateSvdColumns(&diagonalized, &right, 1u, 2u);
    }

    if dot(diagonalized[0], diagonalized[0]) < dot(diagonalized[1], diagonalized[1]) {
        swapSvdColumns(&diagonalized, &right, 0u, 1u);
    }
    if dot(diagonalized[1], diagonalized[1]) < dot(diagonalized[2], diagonalized[2]) {
        swapSvdColumns(&diagonalized, &right, 1u, 2u);
    }
    if dot(diagonalized[0], diagonalized[0]) < dot(diagonalized[1], diagonalized[1]) {
        swapSvdColumns(&diagonalized, &right, 0u, 1u);
    }

    let normalizedSingulars = vec3f(
        length(diagonalized[0]),
        length(diagonalized[1]),
        length(diagonalized[2]),
    );
    let singulars = normalizedSingulars * deformationScale;

    var left0 = vec3f(1.0, 0.0, 0.0);
    if normalizedSingulars.x > SVD_MIN_NORMALIZED_SINGULAR {
        left0 = diagonalized[0] / normalizedSingulars.x;
    }

    let left1Candidate = diagonalized[1] - left0 * dot(left0, diagonalized[1]);
    let left1Length = length(left1Candidate);
    var left1 = perpendicularUnitVector(left0);
    if left1Length > SVD_MIN_NORMALIZED_SINGULAR {
        left1 = left1Candidate / left1Length;
    }

    var left2 = normalize(cross(left0, left1));
    if (
        normalizedSingulars.z > SVD_MIN_NORMALIZED_SINGULAR
        && dot(left2, diagonalized[2]) < 0.0
    ) {
        left2 = -left2;
    }

    return Svd3x3(
        mat3x3f(left0, left1, left2),
        singulars,
        right,
    );
}

/**
 * Projects the elastic singular values into the material's yield interval,
 * then transfers the rejected deformation into F_p while preserving F_e F_p.
 */
fn applyPlasticity(
    particle: ptr<function, ParticleData>,
    material: u32,
) -> bool {
    let deformationElastic = (*particle).deformationElastic;
    let deformationPlastic = (*particle).deformationPlastic;
    if mat3x3IsIdentity(deformationElastic) {
        return false;
    }

    if material == PARTICLE_MATERIAL_BARK || material == PARTICLE_MATERIAL_LEAF {
        return false;
    }

    let stretchMetric = transpose(deformationElastic) * deformationElastic;
    let stretchDelta = stretchMetric - IDENTITY_MAT3;
    let stretchDeltaFrobeniusSquared = dot(stretchDelta[0], stretchDelta[0])
        + dot(stretchDelta[1], stretchDelta[1])
        + dot(stretchDelta[2], stretchDelta[2]);
    if stretchDeltaFrobeniusSquared <= 1e-4 {
        return false;
    }

    var minStretchFactor = 1.0 - DEFAULT_CRITICAL_COMPRESSION;
    var maxStretchFactor = 1.0 + DEFAULT_CRITICAL_STRETCH;
    if material == PARTICLE_MATERIAL_SOIL {
        minStretchFactor = 0.88;
        maxStretchFactor = 1.08;
    } else if material == PARTICLE_MATERIAL_CRUST {
        minStretchFactor = 1.0 - CRUST_CRITICAL_COMPRESSION;
        maxStretchFactor = 1.0 + CRUST_CRITICAL_STRETCH;
    }

    let svd = calculateSvd3x3(deformationElastic);
    if (
        all(vec3f(minStretchFactor) <= svd.singulars)
        && all(svd.singulars <= vec3f(maxStretchFactor))
    ) {
        return false;
    }

    let clampedSingulars = clamp(
        svd.singulars,
        vec3f(minStretchFactor),
        vec3f(maxStretchFactor),
    );
    let newDeformationElastic =
        scaleMatrixColumns(svd.left, clampedSingulars)
        * transpose(svd.right);
    let newDeformationElasticInverse =
        scaleMatrixColumns(svd.right, vec3f(1.0) / clampedSingulars)
        * transpose(svd.left);
    let deformation = deformationElastic * deformationPlastic;

    (*particle).deformationElastic = newDeformationElastic;
    (*particle).deformationPlastic = newDeformationElasticInverse * deformation;
    return true;
}
