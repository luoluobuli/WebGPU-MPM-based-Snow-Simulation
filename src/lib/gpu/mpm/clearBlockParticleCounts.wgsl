@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid : SparseGridStorage;

@compute
@workgroup_size(256)
fn clearBlockParticleCounts(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let thread_index = gid.x;
    let count = atomicLoad(&sparse_grid.n_allocated_blocks);
    if thread_index < min(count, N_MAX_ACTIVE_BLOCKS) {
        atomicStore(&sparse_grid.block_particle_counts[thread_index], 0u);
    }
}
