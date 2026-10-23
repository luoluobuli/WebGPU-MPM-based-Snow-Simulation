const YOUNGS_MODULUS_PA = 1.4e5;
const POISSONS_RATIO = 0.2;

// Lamé parameters
const SHEAR_RESISTANCE = YOUNGS_MODULUS_PA / (2 * (1 + POISSONS_RATIO)); // μ
const VOLUME_RESISTANCE = YOUNGS_MODULUS_PA * POISSONS_RATIO / ((1 + POISSONS_RATIO) * (1 - 2 * POISSONS_RATIO)); // λ
const DEFAULT_HARDENING_COEFFICIENT = 10.0; // ξ
const MIN_HARDENING_SCALE = 0.001;
const MAX_HARDENING_SCALE = 10.0;

// first Piola-Kirchhoff stress tensor
fn calculateStressFirstPiolaKirchhoff( // P
    deformation: mat3x3f, // F
    shearResistance: f32, // μ
    volumetricResistance: f32, // λ
) -> mat3x3f {
    if mat3x3IsIdentity(deformation) {
        return mat3x3f();
    }

    return calculateStressFirstPiolaKirchhoffNonIdentity(
        deformation,
        shearResistance,
        volumetricResistance,
    );
}

fn calculateStressFirstPiolaKirchhoffNonIdentity( // P
    deformation: mat3x3f, // F
    shearResistance: f32, // μ
    volumetricResistance: f32, // λ
) -> mat3x3f {
    let volumeScaleFac = determinant(deformation); // J
    if volumeScaleFac != volumeScaleFac || volumeScaleFac < 0.05 || volumeScaleFac > 20.0 {
        return mat3x3f();
    }

    // corotation to separate out rotation from scaling
    let rotation = calculatePolarDecompositionRotation(deformation); // R

    // P = 2 μ (F - R) + λ (J - 1) J (F⁻¹)ᵀ
    return 2 * shearResistance * (deformation - rotation)
        + volumetricResistance
            * (volumeScaleFac - 1.0)
            * volumeScaleFac
            * transpose(mat3x3InverseWithDeterminant(deformation, volumeScaleFac));
}

// Neo-Hookean constitutive model
fn calculateStressNeoHookean( // P
    deformation: mat3x3f, // F
    shearResistance: f32, // μ
    volumetricResistance: f32, // λ
) -> mat3x3f {
    if mat3x3IsIdentity(deformation) {
        return mat3x3f();
    }

    let volumeScaleFac = determinant(deformation); // J
    if volumeScaleFac != volumeScaleFac || volumeScaleFac <= 0.05 || volumeScaleFac > 20.0 {
        return mat3x3f();
    }

    let deformationInverseTranspose = transpose(mat3x3InverseWithDeterminant(deformation, volumeScaleFac));

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
    if mat3x3IsIdentity(deformationPlastic) {
        return;
    }

    let plastic_volume_scale = determinant(deformationPlastic); // J
    if plastic_volume_scale == 1.0 {
        return;
    }

    let volumeScaleFac = clamp(plastic_volume_scale, 0.2, 5.0);
    let expFac = clamp(
        exp(DEFAULT_HARDENING_COEFFICIENT * (1.0 - volumeScaleFac)),
        MIN_HARDENING_SCALE,
        MAX_HARDENING_SCALE,
    );
    // μ = μ_0 exp(ξ (1 - J))
    // λ = λ_0 exp(ξ (1 - J))
    *baseShearResistance *= expFac;
    *baseVolumetricResistance *= expFac;
}
