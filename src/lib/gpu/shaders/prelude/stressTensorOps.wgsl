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
