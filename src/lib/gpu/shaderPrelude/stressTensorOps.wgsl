const YOUNGS_MODULUS_PA = 1.4e5;
const POISSONS_RATIO = 0.2;

// Lame parameters
const SHEAR_RESISTANCE = YOUNGS_MODULUS_PA / (2 * (1 + POISSONS_RATIO));
const VOLUME_RESISTANCE = YOUNGS_MODULUS_PA * POISSONS_RATIO / ((1 + POISSONS_RATIO) * (1 - 2 * POISSONS_RATIO));

fn calculateStressFirstPiolaKirchhoff(
    deformation: mat3x3f,
    shearResistance: f32,
    volumetricResistance: f32,
) -> mat3x3f {
    let volumeScaleFac = determinant(deformation);
    if volumeScaleFac != volumeScaleFac || volumeScaleFac < 0.05 || volumeScaleFac > 20.0 {
        return mat3x3f();
    }

    let rotation = calculatePolarDecompositionRotation(deformation);
    
    return 2 * shearResistance * (deformation - rotation)
        + volumetricResistance * (volumeScaleFac - 1.0) * volumeScaleFac * transpose(mat3x3Inverse(deformation));
}

fn calculateStressNeoHookean(
    deformation: mat3x3f,
    shearResistance: f32,
    volumetricResistance: f32,
) -> mat3x3f {
    let volumeScaleFac = determinant(deformation);
    if volumeScaleFac != volumeScaleFac || volumeScaleFac <= 0.05 || volumeScaleFac > 20.0 {
        return mat3x3f();
    }

    let deformationInverseTranspose = transpose(mat3x3Inverse(deformation));
    
    return shearResistance * (deformation - deformationInverseTranspose)
        + volumetricResistance * log(volumeScaleFac) * deformationInverseTranspose;
}

fn hardenLameParameters(
    deformationPlastic: mat3x3f,
    baseShearResistance: ptr<function, f32>,
    baseVolumetricResistance: ptr<function, f32>,
) {
    const HARDENING_COEFFICIENT = 5.;

    let volumeScaleFac = clamp(determinant(deformationPlastic), 0.2, 5.0);
    let expFac = clamp(exp(HARDENING_COEFFICIENT * (1.0 - volumeScaleFac)), 0.1, 10.0);
    *baseShearResistance *= expFac;
    *baseVolumetricResistance *= expFac;
}
