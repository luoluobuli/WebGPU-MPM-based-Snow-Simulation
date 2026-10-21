// PBMPM refines a candidate displacement while F_e stays fixed for the step.
// Particle storage retains the shared solver's world-space velocity gradient C,
// so D = dt C on entry and the projected D is converted back to C on exit.
fn solvePbmpmVelocityGradient(
    deformation_elastic: mat3x3f,
    deformation_plastic: mat3x3f,
    velocity_gradient: mat3x3f,
    material: u32,
) -> mat3x3f {
    let timestep = uniforms.simulationTimestep;
    let elastic_det = determinant(deformation_elastic);
    if !(timestep > 0.0)
        || !isFiniteMat3(deformation_elastic)
        || !(elastic_det > 1e-6 && elastic_det < 1e6) {
        return mat3x3f();
    }

    let elastic_inverse = mat3x3InverseWithDeterminant(deformation_elastic, elastic_det);
    let condition_squared = (
        dot(deformation_elastic[0], deformation_elastic[0])
        + dot(deformation_elastic[1], deformation_elastic[1])
        + dot(deformation_elastic[2], deformation_elastic[2])
    ) * (
        dot(elastic_inverse[0], elastic_inverse[0])
        + dot(elastic_inverse[1], elastic_inverse[1])
        + dot(elastic_inverse[2], elastic_inverse[2])
    );
    if !(condition_squared < 1e12) {
        return mat3x3f();
    }

    let displacement_gradient = deformationDeltaFromVelocityGradient(velocity_gradient);
    let trial = (IDENTITY_MAT3 + displacement_gradient) * deformation_elastic;
    let trial_det = determinant(trial);
    if !isFiniteMat3(trial) || !(trial_det > 1e-6 && trial_det < 1e6) {
        return mat3x3f();
    }

    let svd = calculateSvd3x3(trial);
    let rotation = svd.left * transpose(svd.right);
    // Normalize the trial deformation, not its rotation. In 3D the divisor is
    // the cube root of J, making this target's determinant one.
    let volume_target = trial * pow(trial_det, -1.0 / 3.0);

    var shear_resistance = SHEAR_RESISTANCE;
    var volume_resistance = VOLUME_RESISTANCE;
    applyMaterialLameParameters(material, &shear_resistance, &volume_resistance);
    hardenLameParameters(
        material,
        deformation_plastic,
        &shear_resistance,
        &volume_resistance,
    );
    let bulk_resistance = volume_resistance + 2.0 * shear_resistance / 3.0;
    let shape_weight = shear_resistance / max(shear_resistance + bulk_resistance, 1e-6);
    let target_deformation = shape_weight * rotation + (1.0 - shape_weight) * volume_target;
    let target_displacement = target_deformation * elastic_inverse - IDENTITY_MAT3;

    // Bounded local relaxation preserves material/hardening differences and
    // tends to a rigid projection as dt * wave_speed / cell_size grows.
    let inverse_cell_size = max(
        max(uniforms.invGridCellDims.x, uniforms.invGridCellDims.y),
        uniforms.invGridCellDims.z,
    );
    let stiffness = min(
        (volume_resistance + 2.0 * shear_resistance)
            / 400.0
            * timestep * timestep
            * inverse_cell_size * inverse_cell_size,
        1e6,
    );
    let relaxation = stiffness / (1.0 + stiffness);
    let corrected_displacement = displacement_gradient
        + relaxation * (target_displacement - displacement_gradient);
    return sanitizeVelocityGradient(corrected_displacement * uniforms.invSimulationTimestep);
}
