@group(1) @binding(0) var<storage, read_write> hash_map_entries: array<HashMapEntry>;
@group(1) @binding(1) var<storage, read_write> n_allocated_blocks: atomic<u32>;
// @group(1) @binding(7) var<storage, read_write> indirect_dispatch: array<u32>;

@compute
@workgroup_size(256)
fn clearHashMap(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let thread_index = gid.x;
    
    if thread_index == 0u {
        atomicStore(&n_allocated_blocks, 0u);
    }

    if thread_index >= arrayLength(&hash_map_entries) { return; }

    atomicStore(&hash_map_entries[thread_index].block_index, GRID_HASH_MAP_BLOCK_INDEX_EMPTY);
    // hash_map_entries[thread_index].block_number = vec3i(0, 0, 0);
}
