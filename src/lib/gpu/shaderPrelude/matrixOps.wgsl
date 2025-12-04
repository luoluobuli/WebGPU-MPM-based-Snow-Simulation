const IDENTITY_MAT3 = mat3x3f(
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
);


fn mat3x3Inverse(matrix: mat3x3f) -> mat3x3f {
    let det = determinant(matrix);
    if det == 0 {
        return IDENTITY_MAT3;
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


/**
 * iterative polar decomposition approximates pure rotation R from F where F = R S
 */
fn calculatePolarDecompositionRotation(deformation: mat3x3f) -> mat3x3f {
    var rotationGuess = deformation; // R_n
    
    const N_ITERATIONS = 4;
    for (var i = 0u; i < N_ITERATIONS; i++) {
        // idea: the closer R_n gets to the true rotation R, the closer the inverse transpose should be to R as well
        let rotationGuessInverseTranspose = transpose(mat3x3Inverse(rotationGuess));
        // so average toward it
        rotationGuess = 0.5 * (rotationGuess + rotationGuessInverseTranspose);
    }
    
    return rotationGuess;
}

fn outerProduct(u: vec3f, v: vec3f) -> mat3x3f {
    return mat3x3f(
        u * v.x,
        u * v.y,
        u * v.z,
    );
}
