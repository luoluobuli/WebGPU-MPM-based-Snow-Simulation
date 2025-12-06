@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@compute
@workgroup_size(256)
fn clearHashMap(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let thread_index = gid.x;
    
    if thread_index == 0u {
        atomicStore(&sparse_grid.n_allocated_blocks, 0u);
    }

    if thread_index >= HASH_MAP_SIZE { return; }

    atomicStore(&sparse_grid.hash_map_entries[thread_index].block_index, GRID_HASH_MAP_BLOCK_INDEX_EMPTY);
}
