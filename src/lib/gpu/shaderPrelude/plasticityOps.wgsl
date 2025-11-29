/**
 * Checks whether a particle's elastic deformation indicates that the particle should undergo permanent (plastic)
 * deformation. This check is done by comparing the amount of stretching (non-uniform scaling) to a minimum stretch
 * factor and maximum stretch factor. If an actual stretch factor in some direction lies outside of this range, the
 * deformation matrices are updated to use a clamped amount of stretching in that direction.
 */
fn applyPlasticity(
    particle: ptr<function, ParticleData>
) {
    var dE = (*particle).deformationElastic; // F_e
    var dP = (*particle).deformationPlastic; // F_p
    // extract rotation and scale from the elastic deformation
    let rotation = calculatePolarDecompositionRotation(dE); // R
    // S = Rᵀ F_e  (<=  F_e = R S) 
    let scale = transpose(rotation) * (dE); // S
    
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
    let deformation = (dE) * (dP);
    
    (*particle).deformationElastic = newDeformationElastic;
    // F_p = (F_e)⁻¹ F  (<=  F = F_e F_p)  
    (*particle).deformationPlastic = newDeformationElasticInv * deformation;
}
