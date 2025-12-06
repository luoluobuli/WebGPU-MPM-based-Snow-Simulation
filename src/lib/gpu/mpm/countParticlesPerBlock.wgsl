@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> hash_map_entries: array<HashMapEntry>;
@group(1) @binding(7) var<storage, read_write> block_particle_counts: array<atomic<u32>>;

@group(2) @binding(0) var<storage, read_write> particle_data: array<ParticleData>;

@compute
@workgroup_size(256)
fn countParticlesPerBlock(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let thread_index = gid.x;
    if thread_index >= arrayLength(&particle_data) { return; }

    let particle = particle_data[thread_index];
    let cell_number = calculateCellNumber(particle.pos);
    let block_number = calculateBlockNumberContainingCell(cell_number);
    let block_index = retrieveBlockIndexFromHashMap(block_number);

    if block_index != GRID_HASH_MAP_BLOCK_INDEX_EMPTY {
        atomicAdd(&block_particle_counts[block_index], 1u);
    }
}
