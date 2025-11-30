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

    
    // PBMPM Affine Correction
    // Apply viscosity and volume preservation to D before transfer
    var D = particle.deformation_displacement;
    
    // Viscosity
    let viscosity = 1.0; // Hardcoded for now, could be a uniform
    let deviatoric = -1.0 * (D + transpose(D));
    D += viscosity * 0.5 * deviatoric;

    // Volume Preservation
    let J = determinant(particle.deformationElastic);
    // trace(D)
    let traceD = D[0][0] + D[1][1] + D[2][2];
    let alpha = 0.5 * (1.0 / (J + 1e-3) - traceD - 1.0);
    let volumeRelax = 0.2; // Hardcoded for now
    
    // Add alpha * Identity
    D[0][0] += volumeRelax * alpha;
    D[1][1] += volumeRelax * alpha;
    D[2][2] += volumeRelax * alpha;

    particle.deformation_displacement = D;


    // let rotation = calculatePolarDecompositionRotation(particle.deformationElastic);



    // // inverse of the formula used to integrate deformation
    // let candidate_deformation_displacement = rotation * mat3x3Inverse(particle.deformationElastic) - mat3x3Identity();


    // let elasticity_relaxation = 0.5;
    // particle.deformation_displacement = particle.deformation_displacement * elasticity_relaxation
    //     + candidate_deformation_displacement * (1 - elasticity_relaxation);




    particle_data[particle_index] = particle;
}