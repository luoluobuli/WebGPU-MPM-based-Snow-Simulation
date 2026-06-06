const YOUNGS_MODULUS_PA = 1.4e5;
const POISSONS_RATIO = 0.2;

// Lame parameters
const SHEAR_RESISTANCE = YOUNGS_MODULUS_PA / (2 * (1 + POISSONS_RATIO));
const VOLUME_RESISTANCE = YOUNGS_MODULUS_PA * POISSONS_RATIO / ((1 + POISSONS_RATIO) * (1 - 2 * POISSONS_RATIO));
const SOIL_SHEAR_RESISTANCE_SCALE = 0.1;
const SOIL_VOLUME_RESISTANCE_SCALE = 0.14;
const BARK_SHEAR_RESISTANCE_SCALE = 6.0;
const BARK_VOLUME_RESISTANCE_SCALE = 8.0;
const LEAF_SHEAR_RESISTANCE_SCALE = 0.14;
const LEAF_VOLUME_RESISTANCE_SCALE = 0.18;

fn applyMaterialLameParameters(
    material: u32,
    shearResistance: ptr<function, f32>,
    volumetricResistance: ptr<function, f32>,
) {
    if material == PARTICLE_MATERIAL_SOIL {
        *shearResistance = SHEAR_RESISTANCE * SOIL_SHEAR_RESISTANCE_SCALE;
        *volumetricResistance = VOLUME_RESISTANCE * SOIL_VOLUME_RESISTANCE_SCALE;
    } else if material == PARTICLE_MATERIAL_BARK {
        *shearResistance = SHEAR_RESISTANCE * BARK_SHEAR_RESISTANCE_SCALE;
        *volumetricResistance = VOLUME_RESISTANCE * BARK_VOLUME_RESISTANCE_SCALE;
    } else if material == PARTICLE_MATERIAL_LEAF {
        *shearResistance = SHEAR_RESISTANCE * LEAF_SHEAR_RESISTANCE_SCALE;
        *volumetricResistance = VOLUME_RESISTANCE * LEAF_VOLUME_RESISTANCE_SCALE;
    }
}

fn calculateStressFirstPiolaKirchhoff(
    deformation: mat3x3f,
    shearResistance: f32,
    volumetricResistance: f32,
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

fn calculateStressFirstPiolaKirchhoffNonIdentity(
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
        + volumetricResistance
            * (volumeScaleFac - 1.0)
            * volumeScaleFac
            * transpose(mat3x3InverseWithDeterminant(deformation, volumeScaleFac));
}

fn calculateStressNeoHookean(
    deformation: mat3x3f,
    shearResistance: f32,
    volumetricResistance: f32,
) -> mat3x3f {
    if mat3x3IsIdentity(deformation) {
        return mat3x3f();
    }

    let volumeScaleFac = determinant(deformation);
    if volumeScaleFac != volumeScaleFac || volumeScaleFac <= 0.05 || volumeScaleFac > 20.0 {
        return mat3x3f();
    }

    let deformationInverseTranspose = transpose(mat3x3InverseWithDeterminant(deformation, volumeScaleFac));

    return shearResistance * (deformation - deformationInverseTranspose)
        + volumetricResistance * log(volumeScaleFac) * deformationInverseTranspose;
}

fn hardenLameParameters(
    material: u32,
    deformationPlastic: mat3x3f,
    baseShearResistance: ptr<function, f32>,
    baseVolumetricResistance: ptr<function, f32>,
) {
    const HARDENING_COEFFICIENT = 5.;

    if material == PARTICLE_MATERIAL_BARK || material == PARTICLE_MATERIAL_LEAF {
        return;
    }

    if mat3x3IsIdentity(deformationPlastic) {
        return;
    }

    let plastic_volume_scale = determinant(deformationPlastic);
    if plastic_volume_scale == 1.0 {
        return;
    }

    let volumeScaleFac = clamp(plastic_volume_scale, 0.2, 5.0);
    let expFac = clamp(exp(HARDENING_COEFFICIENT * (1.0 - volumeScaleFac)), 0.1, 10.0);
    *baseShearResistance *= expFac;
    *baseVolumetricResistance *= expFac;
}
