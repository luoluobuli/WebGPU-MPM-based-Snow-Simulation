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

    let particle = &particle_data[particle_index];
}