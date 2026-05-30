@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid : SparseGridStorage;

@group(2) @binding(0) var<storage, read_write> particleData: array<ParticleData>;

fn allocateBlock(block_number: vec3i) {
    if !bukkitCanContainBlock(block_number) {
        return;
    }

    let bukkit_index = calculateBukkitIndex(block_number);

    for (var i = 0u; i < 64u; i++) {
        var current_allocated_block_index = atomicLoad(&sparse_grid.block_index_bukkits[bukkit_index]);

        if current_allocated_block_index == GRID_BLOCK_INDEX_EMPTY {
            let res = atomicCompareExchangeWeak(
                &sparse_grid.block_index_bukkits[bukkit_index],
                GRID_BLOCK_INDEX_EMPTY,
                GRID_BLOCK_INDEX_RESERVED
            );

            if res.exchanged {
                let next_block_index = atomicAdd(&sparse_grid.n_allocated_blocks, 1u);

                if next_block_index >= N_MAX_ACTIVE_BLOCKS {
                    atomicStore(&sparse_grid.block_index_bukkits[bukkit_index], GRID_BLOCK_INDEX_EMPTY);
                    return; 
                }

                sparse_grid.mapped_block_numbers[next_block_index] = vec4i(block_number, 0i);
                atomicStore(&sparse_grid.block_index_bukkits[bukkit_index], next_block_index);
                return;
            }

            continue;
        }

        if current_allocated_block_index != GRID_BLOCK_INDEX_RESERVED {
            return;
        }

        // Another invocation is assigning this bukkit. Spin briefly so later
        // dispatches see a stable block index without paying hash probes.
    }
}

@compute
@workgroup_size(256)
fn mapAffectedBlocks(@builtin(global_invocation_id) gid: vec3u) {
    let threadIndex = gid.x;
    if threadIndex >= arrayLength(&particleData) { return; }

    let particle = particleData[threadIndex];
    if !particlePositionCanTouchGrid(particle.pos) { return; }
    
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
