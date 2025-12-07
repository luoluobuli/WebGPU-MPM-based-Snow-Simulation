@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid : SparseGridStorage;

@compute
@workgroup_size(256)
fn clearBlockParticleCounts(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let thread_index = gid.x;
    if thread_index < N_MAX_BLOCKS_IN_HASH_MAP {
        atomicStore(&sparse_grid.block_particle_counts[thread_index], 0u);
    }
}
