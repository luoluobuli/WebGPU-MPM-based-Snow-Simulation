@group(1) @binding(0) var<storage, read_write> hash_map_entries: array<HashMapEntry>;
@group(1) @binding(1) var<storage, read_write> n_allocated_blocks: atomic<u32>;
@group(1) @binding(2) var<storage, read_write> mapped_block_indexes: array<u32>;
@group(2) @binding(0) var<storage, read_write> particleData: array<ParticleData>;

fn allocateBlock(block_number: vec3i) {
    let hashed_index = hash3(bitcast<vec3u>(block_number));
    var candidate_hash_map_index = hashed_index % HASH_MAP_SIZE;
    
    for (var i = 0u; i < N_HASH_MAP_CANDIDATE_INDEX_ATTEMPTS; i++) {
        // we'll try to insert something into the hash table at this index
        
        // check what's in the hash table here...
        var current_allocated_block_index = atomicLoad(&hash_map_entries[candidate_hash_map_index].block_index);
        
        // is this index empty?
        if current_allocated_block_index == GRID_HASH_MAP_BLOCK_INDEX_EMPTY {
            // try to reserve it
            let res = atomicCompareExchangeWeak(&hash_map_entries[candidate_hash_map_index].block_index, GRID_HASH_MAP_BLOCK_INDEX_EMPTY, GRID_HASH_MAP_BLOCK_INDEX_RESERVED);
            
            // did we get the reservation?
            if res.exchanged {
                let next_block_index = atomicAdd(&n_allocated_blocks, 1u);
                
                // is there still space in the map?
                if next_block_index >= N_MAX_BLOCKS_IN_HASH_MAP {
                    // TODO should this be the empty key?
                    atomicStore(&hash_map_entries[candidate_hash_map_index].block_index, GRID_HASH_MAP_BLOCK_INDEX_EMPTY); 
                    return; 
                }
                
                // everything is good! store the block number
                hash_map_entries[candidate_hash_map_index].block_number = block_number; // mark this block as ours
                mapped_block_indexes[next_block_index] = candidate_hash_map_index; // we'll want go from block index to hashmap index later
                atomicStore(&hash_map_entries[candidate_hash_map_index].block_index, next_block_index); // make the block index accessible from the hashmap
                return;
            }

            // we didn't get it. reread
            current_allocated_block_index = atomicLoad(&hash_map_entries[candidate_hash_map_index].block_index);
        }
        
        // if the index is reserved, wait for it to free up (spin loop lol)
        var n_spin_loop_iterations = 0u;
        while current_allocated_block_index == GRID_HASH_MAP_BLOCK_INDEX_RESERVED && n_spin_loop_iterations < 32 {
            current_allocated_block_index = atomicLoad(&hash_map_entries[candidate_hash_map_index].block_index);
            n_spin_loop_iterations++;
        }
        
        // if it's still empty or reserved, allocation failed :( probe another index
        if current_allocated_block_index == GRID_HASH_MAP_BLOCK_INDEX_EMPTY || current_allocated_block_index == GRID_HASH_MAP_BLOCK_INDEX_RESERVED { continue; }

        let block_number_in_page_table = hash_map_entries[candidate_hash_map_index].block_number;

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
    
    // allocate every cell that this particle is going to affect in the p2g step
    for (var offset_z = -1i; offset_z <= 1; offset_z++) {
        for (var offset_y = -1i; offset_y <= 1; offset_y++) {
            for (var offset_x = -1i; offset_x <= 1; offset_x++) {
                let cell_number = start_cell_number + vec3i(offset_x, offset_y, offset_z);
                if !cellNumberInGridRange(cell_number) { continue; }
                
                let block_number = calculateBlockNumberContainingCell(cell_number);
                allocateBlock(block_number);
            }
        }
    }
}