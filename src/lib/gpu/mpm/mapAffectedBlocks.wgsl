@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid : SparseGridStorage;

@group(2) @binding(0) var<storage, read_write> particleData: array<ParticleData>;

fn allocateBlock(block_number: vec3i) {
    let hashed_index = hash3(bitcast<vec3u>(block_number));
    var candidate_hash_map_index = hashed_index % HASH_MAP_SIZE;
    
    for (var i = 0u; i < N_HASH_MAP_CANDIDATE_INDEX_ATTEMPTS; i++) {
        // we'll try to insert something into the hash table at this index
        
        // check what's in the hash table here...
        var current_allocated_block_index = atomicLoad(&sparse_grid.hash_map_entries[candidate_hash_map_index].block_index);
        
        // is this index empty?
        if current_allocated_block_index == GRID_HASH_MAP_BLOCK_INDEX_EMPTY {
            // try to reserve it
            let res = atomicCompareExchangeWeak(&sparse_grid.hash_map_entries[candidate_hash_map_index].block_index, GRID_HASH_MAP_BLOCK_INDEX_EMPTY, GRID_HASH_MAP_BLOCK_INDEX_RESERVED);
            
            // did we get the reservation?
            if res.exchanged {
                let next_block_index = atomicAdd(&sparse_grid.n_allocated_blocks, 1u);
                
                // is there still space in the map?
                if next_block_index >= N_MAX_BLOCKS_IN_HASH_MAP {
                    // TODO should this be the empty key?
                    atomicStore(&sparse_grid.hash_map_entries[candidate_hash_map_index].block_index, GRID_HASH_MAP_BLOCK_INDEX_EMPTY); 
                    return; 
                }
                
                // everything is good! store the block number
                sparse_grid.hash_map_entries[candidate_hash_map_index].block_number = block_number; // mark this block as ours
                sparse_grid.mapped_block_indexes[next_block_index] = candidate_hash_map_index; // we'll want go from block index to hashmap index later
                atomicStore(&sparse_grid.hash_map_entries[candidate_hash_map_index].block_index, next_block_index); // make the block index accessible from the hashmap
                return;
            }

            // we didn't get it. reread
            current_allocated_block_index = atomicLoad(&sparse_grid.hash_map_entries[candidate_hash_map_index].block_index);
        }
        
        // if the index is reserved, wait for it to free up (spin loop lol)
        var n_spin_loop_iterations = 0u;
        while current_allocated_block_index == GRID_HASH_MAP_BLOCK_INDEX_RESERVED && n_spin_loop_iterations < 32 {
            current_allocated_block_index = atomicLoad(&sparse_grid.hash_map_entries[candidate_hash_map_index].block_index);
            n_spin_loop_iterations++;
        }
        
        // if it's still empty or reserved, allocation failed :( probe another index
        if current_allocated_block_index == GRID_HASH_MAP_BLOCK_INDEX_EMPTY || current_allocated_block_index == GRID_HASH_MAP_BLOCK_INDEX_RESERVED { continue; }

        let block_number_in_page_table = sparse_grid.hash_map_entries[candidate_hash_map_index].block_number;

        // we already got this block
        if all(block_number_in_page_table == block_number) { return; }

        // there's already another block here, probe another index
        candidate_hash_map_index = select(candidate_hash_map_index + 1, 0, candidate_hash_map_index + 1 >= HASH_MAP_SIZE);
    }
}

@compute
@workgroup_size(256)
fn mapAffectedBlocks(@builtin(global_invocation_id) gid: vec3u) {
    let threadIndex = gid.x;
    if threadIndex >= arrayLength(&particleData) { return; }

    let particle = particleData[threadIndex];
    
    let start_cell_number = calculateCellNumber(particle.pos);

    let last_grid_cell = vec3i(uniforms.gridResolution) - vec3i(1);
    let min_cell = max(start_cell_number - vec3i(1), vec3i(0));
    let max_cell = min(start_cell_number + vec3i(1), last_grid_cell);
    if any(max_cell < min_cell) { return; }

    let min_block = calculateBlockNumberContainingCell(min_cell);
    let max_block = calculateBlockNumberContainingCell(max_cell);

    for (var block_z = min_block.z; block_z <= max_block.z; block_z++) {
        for (var block_y = min_block.y; block_y <= max_block.y; block_y++) {
            for (var block_x = min_block.x; block_x <= max_block.x; block_x++) {
                allocateBlock(vec3i(block_x, block_y, block_z));
            }
        }
    }
}
