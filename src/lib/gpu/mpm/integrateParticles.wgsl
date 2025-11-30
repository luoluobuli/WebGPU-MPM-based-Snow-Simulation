@group(1) @binding(0) var<storage, read_write> hash_map_entries: array<HashMapEntry>;
@group(1) @binding(1) var<storage, read_write> n_allocated_blocks: atomic<u32>;
@group(1) @binding(2) var<storage, read_write> mapped_block_indexes: array<u32>;

@group(2) @binding(0) var<storage, read_write> particle_data: array<ParticleData>;

@compute
@workgroup_size(256)
fn integrateParticles(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let particle_index = gid.x;
    if particle_index > arrayLength(&particle_data) { return; }

    var particle = particle_data[particle_index];

    // let gravitational_acceleration = vec3f(0, 0, -9.81);
    // particle.pos_displacement += gravitational_acceleration * uniforms.simulationTimestep * uniforms.simulationTimestep;

    particle.pos += particle.pos_displacement;
    particle.deformationElastic = (mat3x3Identity() + particle.deformation_displacement) * particle.deformationElastic;

    applyPlasticity(&particle);
    
    // Boundary conditions
    if particle.pos.x < uniforms.gridMinCoords.x {
        particle.pos_displacement.x *= -0.5;
        particle.pos.x = uniforms.gridMinCoords.x;
    }
    if particle.pos.x >= uniforms.gridMaxCoords.x {
        particle.pos_displacement.x *= -0.5;
        particle.pos.x = uniforms.gridMaxCoords.x;
    }

    if particle.pos.y < uniforms.gridMinCoords.y {
        particle.pos_displacement.y *= -0.5;
        particle.pos.y = uniforms.gridMinCoords.y;
    }
    if particle.pos.y >= uniforms.gridMaxCoords.y {
        particle.pos_displacement.y *= -0.5;
        particle.pos.y = uniforms.gridMaxCoords.y;
    }

    if particle.pos.z < uniforms.gridMinCoords.z {
        particle.pos_displacement.z *= -0.5;
        particle.pos.z = uniforms.gridMinCoords.z;
    }
    if particle.pos.z >= uniforms.gridMaxCoords.z {
        particle.pos_displacement.z *= -0.5;
        particle.pos.z = uniforms.gridMaxCoords.z;
    }

    particle_data[particle_index] = particle;
}