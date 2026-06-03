@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid : SparseGridStorage;

@group(2) @binding(0) var<storage, read> particleData: array<ParticleData>;

override N_PARTICLES: u32 = 0u;

@compute
@workgroup_size(256)
fn mapAffectedBlocks(@builtin(global_invocation_id) gid: vec3u) {
    let threadIndex = gid.x;
    if threadIndex >= N_PARTICLES { return; }

    let particle_pos = particleData[threadIndex].pos;
    mapParticleAffectedBlocks(particle_pos, sparse_grid.current_generation);
}
