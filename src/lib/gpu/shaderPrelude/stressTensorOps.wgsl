const YOUNGS_MODULUS_PA = 1.4e5;
const POISSONS_RATIO = 0.2;

// Lamé parameters
const SHEAR_RESISTANCE = YOUNGS_MODULUS_PA / (2 * (1 + POISSONS_RATIO)); // μ
const VOLUME_RESISTANCE = YOUNGS_MODULUS_PA * POISSONS_RATIO / ((1 + POISSONS_RATIO) * (1 - 2 * POISSONS_RATIO)); // λ
    
// first Piola-Kirchhoff stress tensor
fn calculateStressFirstPiolaKirchhoff( // P
    deformation: mat3x3f, // F
    shearResistance: f32, // μ
    volumetricResistance: f32, // λ
) -> mat3x3f {
    let volumeScaleFac = determinant(deformation); // J

    // corotation to separate out rotation from scaling
    let rotation = calculatePolarDecompositionRotation(deformation); // R
    
    // P = 2 μ (F - R) + λ (J - 1) J (F⁻¹)ᵀ
    return 2 * shearResistance * (deformation - rotation)
        + volumetricResistance * (volumeScaleFac - 1) * volumeScaleFac * transpose(mat3x3Inverse(deformation));
}

// Neo-Hookean constitutive model
fn calculateStressNeoHookean( // P
    deformation: mat3x3f, // F
    shearResistance: f32, // μ
    volumetricResistance: f32, // λ
) -> mat3x3f {
    let volumeScaleFac = determinant(deformation); // J
    let deformationInverseTranspose = transpose(mat3x3Inverse(deformation));
    
    // P = μ (F - (F⁻¹)ᵀ) + λ log(J) (F⁻¹)ᵀ
    return shearResistance * (deformation - deformationInverseTranspose)
        + volumetricResistance * log(volumeScaleFac) * deformationInverseTranspose;
}

// from Stomakhin 
fn hardenLameParameters(
    deformationPlastic: mat3x3f, // F_p
    baseShearResistance: ptr<function, f32>, // μ_0
    baseVolumetricResistance: ptr<function, f32>, // λ_0
) {
    const HARDENING_COEFFICIENT = 5.; // ξ

    let volumeScaleFac = determinant(deformationPlastic); // J

    // check if the particle is being compressed
    // if volumeScaleFac >= 1 { return; }
    
    // μ = μ_0 exp(ξ (1 - J))
    // λ = λ_0 exp(ξ (1 - J))
    let expFac = exp(HARDENING_COEFFICIENT * (1 - volumeScaleFac));
    *baseShearResistance *= expFac;
    *baseVolumetricResistance *= expFac;
}