@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(2) @binding(0) var<storage, read_write> particle_data: array<ParticleData>;
@group(2) @binding(1) var<storage, read_write> sorted_particle_indices: array<u32>;

@compute
@workgroup_size(256)
fn binParticles(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let thread_index = gid.x;
    if thread_index >= arrayLength(&particle_data) { return; }

    let particle = particle_data[thread_index];
    let cell_number = calculateCellNumber(particle.pos);
    let block_number = calculateBlockNumberContainingCell(cell_number);
    let block_index = retrieveBlockIndexFromHashMap(block_number);

    if block_index != GRID_HASH_MAP_BLOCK_INDEX_EMPTY {
        let dest_index = atomicAdd(&sparse_grid.block_particle_offsets[block_index], 1u);
        sorted_particle_indices[dest_index] = thread_index;
    }
}
