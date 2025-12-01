@group(1) @binding(0) var<storage, read_write> hash_map_entries: array<HashMapEntry>;
@group(1) @binding(1) var<storage, read_write> n_allocated_blocks: atomic<u32>;
@group(1) @binding(2) var<storage, read_write> mapped_block_indexes: array<u32>;

@group(2) @binding(0) var<storage, read_write> particle_data: array<ParticleData>;

@compute
@workgroup_size(256)
fn solveParticleConstraints(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let particle_index = gid.x;
    if particle_index > arrayLength(&particle_data) { return; }

    var particle = particle_data[particle_index];

    // this is the inverse of the formula used to integrate deformation
    let trial_deformation_elastic = (mat3x3Identity() + particle.deformation_displacement) * particle.deformationElastic; 
    let trial_rotation = calculatePolarDecompositionRotation(trial_deformation_elastic);

    let target_volume = determinant(trial_deformation_elastic);
    
    let volume_scale = pow(abs(target_volume), -0.3333333);
    let target_scaled = trial_rotation * volume_scale;

    let blend_factor = 0.95;
    let target_blended = blend_factor * target_scaled + (1 - blend_factor) * trial_rotation;

    let corrected_deformation_displacement = target_blended * mat3x3Inverse(particle.deformationElastic) - mat3x3Identity();
    
    let deformation_displacement_diff = corrected_deformation_displacement - particle.deformation_displacement;
    let elasticity_relaxation = 0.95; 
    particle.deformation_displacement += elasticity_relaxation * deformation_displacement_diff;

    particle_data[particle_index] = particle;
}