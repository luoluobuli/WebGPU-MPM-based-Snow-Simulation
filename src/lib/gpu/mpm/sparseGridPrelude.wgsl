const BLOCK_SIZE = 4u;
const BLOCK_SIZE_CUBED = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
const LOG_BLOCK_SIZE = 2u; // log2
const LOG_BLOCK_SIZE_CUBED = LOG_BLOCK_SIZE * 3u;
const BLOCK_MASK = 3u; // 4 - 1

const BUKKIT_DOMAIN_GRID_RESOLUTION = 384u;
const BUKKIT_DOMAIN_BLOCKS_X = (BUKKIT_DOMAIN_GRID_RESOLUTION + BLOCK_SIZE - 1u) / BLOCK_SIZE;
const BUKKIT_DOMAIN_BLOCKS_Y = BUKKIT_DOMAIN_BLOCKS_X;
const BUKKIT_DOMAIN_BLOCKS_Z = BUKKIT_DOMAIN_BLOCKS_X;
const BUKKIT_DOMAIN_MAX_BLOCK = vec3i(
    i32(BUKKIT_DOMAIN_BLOCKS_X) - 1i,
    i32(BUKKIT_DOMAIN_BLOCKS_Y) - 1i,
    i32(BUKKIT_DOMAIN_BLOCKS_Z) - 1i,
);
const BUKKIT_DOMAIN_BLOCK_COUNT = BUKKIT_DOMAIN_BLOCKS_X * BUKKIT_DOMAIN_BLOCKS_Y * BUKKIT_DOMAIN_BLOCKS_Z;

const N_MAX_ACTIVE_BLOCKS = 100000u;
const GRID_BLOCK_INDEX_EMPTY = 0xFFFFFFFFu;
const GRID_BLOCK_INDEX_RESERVED = 0xFFFFFFFEu;
const BUKKIT_GENERATION_RESERVED = 0xFFFFFFFFu;

override GRID_LAST_CELL_X: i32 = 383i;
override GRID_LAST_CELL_Y: i32 = 383i;
override GRID_LAST_CELL_Z: i32 = 383i;
override GRID_DOMAIN_MAX_BLOCK_X: i32 = 95i;
override GRID_DOMAIN_MAX_BLOCK_Y: i32 = 95i;
override GRID_DOMAIN_MAX_BLOCK_Z: i32 = 95i;

struct SparseGridStorage {
    n_allocated_blocks: atomic<u32>,
    current_generation: u32,
    _padding: array<u32, 2>,
    block_index_bukkits: array<atomic<u32>, BUKKIT_DOMAIN_BLOCK_COUNT>,
    bukkit_generations: array<atomic<u32>, BUKKIT_DOMAIN_BLOCK_COUNT>,
    mapped_block_numbers: array<vec3i, N_MAX_ACTIVE_BLOCKS>,
}

struct BlockNeighborhood {
    min_block: vec3i,
    block_indices: array<u32, 8>,
    single_block_cell_index_base: u32,
    check_cell_range: u32,
}

fn calculateBlockNumberContainingCell(cell_number: vec3i) -> vec3i {
    return cell_number >> vec3u(LOG_BLOCK_SIZE);
}

fn calculateCellIndexWithinBlock(cell_number: vec3i) -> u32 {
    let cell_index_within_block = vec3u(cell_number & vec3i(i32(BLOCK_MASK)));
    return cell_index_within_block.x
        | (cell_index_within_block.y << LOG_BLOCK_SIZE)
        | (cell_index_within_block.z << (LOG_BLOCK_SIZE * 2u));
}

fn gridLastCell() -> vec3i {
    return vec3i(GRID_LAST_CELL_X, GRID_LAST_CELL_Y, GRID_LAST_CELL_Z);
}

fn gridDomainMaxBlock() -> vec3i {
    return vec3i(GRID_DOMAIN_MAX_BLOCK_X, GRID_DOMAIN_MAX_BLOCK_Y, GRID_DOMAIN_MAX_BLOCK_Z);
}

fn cellNumberInSparseGridRange(cell_number: vec3i) -> bool {
    return all(vec3i(0) <= cell_number) && all(cell_number <= gridLastCell());
}

fn bukkitCanContainBlock(block_number: vec3i) -> bool {
    return all(block_number >= vec3i(0))
        && all(block_number <= gridDomainMaxBlock());
}

fn calculateBukkitIndex(block_number: vec3i) -> u32 {
    let block = vec3u(block_number);
    return block.x + BUKKIT_DOMAIN_BLOCKS_X * (block.y + BUKKIT_DOMAIN_BLOCKS_Y * block.z);
}

fn allocateBlockInDomain(block_number: vec3i, target_generation: u32) {
    let bukkit_index = calculateBukkitIndex(block_number);

    for (var i = 0u; i < 64u; i++) {
        let current_generation = atomicLoad(&sparse_grid.bukkit_generations[bukkit_index]);

        if current_generation != target_generation {
            if current_generation != BUKKIT_GENERATION_RESERVED {
                let res = atomicCompareExchangeWeak(
                    &sparse_grid.bukkit_generations[bukkit_index],
                    current_generation,
                    BUKKIT_GENERATION_RESERVED
                );

                if res.exchanged {
                    let next_block_index = atomicAdd(&sparse_grid.n_allocated_blocks, 1u);

                    if next_block_index >= N_MAX_ACTIVE_BLOCKS {
                        atomicStore(&sparse_grid.block_index_bukkits[bukkit_index], GRID_BLOCK_INDEX_EMPTY);
                        atomicStore(&sparse_grid.bukkit_generations[bukkit_index], target_generation);
                        return;
                    }

                    sparse_grid.mapped_block_numbers[next_block_index] = block_number;
                    atomicStore(&sparse_grid.block_index_bukkits[bukkit_index], next_block_index);
                    atomicStore(&sparse_grid.bukkit_generations[bukkit_index], target_generation);
                    return;
                }
            }

            continue;
        }

        let current_allocated_block_index = atomicLoad(&sparse_grid.block_index_bukkits[bukkit_index]);
        if current_allocated_block_index != GRID_BLOCK_INDEX_RESERVED {
            return;
        }

        // Another invocation is assigning this bukkit. Spin briefly so later
        // dispatches see a stable block index without paying hash probes.
    }
}

fn mapParticleAffectedBlocksInGrid(particle_pos: vec3f, target_generation: u32) {
    let start_cell_number = calculateCellNumber(particle_pos);

    let start_block = calculateBlockNumberContainingCell(start_cell_number);
    let cell_offset = start_cell_number & vec3i(i32(BLOCK_MASK));
    if all(cell_offset > vec3i(0)) && all(cell_offset < vec3i(i32(BLOCK_MASK))) {
        allocateBlockInDomain(start_block, target_generation);
        return;
    }

    let lower_block_delta = vec3i(
        select(0i, 1i, cell_offset.x == 0i),
        select(0i, 1i, cell_offset.y == 0i),
        select(0i, 1i, cell_offset.z == 0i),
    );
    let upper_block_delta = vec3i(
        select(0i, 1i, cell_offset.x == i32(BLOCK_MASK)),
        select(0i, 1i, cell_offset.y == i32(BLOCK_MASK)),
        select(0i, 1i, cell_offset.z == i32(BLOCK_MASK)),
    );
    let min_block = max(start_block - lower_block_delta, vec3i(0));
    let max_block = min(start_block + upper_block_delta, gridDomainMaxBlock());

    if all(min_block == max_block) {
        allocateBlockInDomain(min_block, target_generation);
        return;
    }

    for (var block_z = min_block.z; block_z <= max_block.z; block_z++) {
        for (var block_y = min_block.y; block_y <= max_block.y; block_y++) {
            for (var block_x = min_block.x; block_x <= max_block.x; block_x++) {
                allocateBlockInDomain(vec3i(block_x, block_y, block_z), target_generation);
            }
        }
    }
}

fn mapParticleAffectedBlocks(particle_pos: vec3f, target_generation: u32) {
    if !particlePositionCanTouchGrid(particle_pos) { return; }

    mapParticleAffectedBlocksInGrid(particle_pos, target_generation);
}

fn retrieveBlockIndexFromBukkitGeneration(block_number: vec3i, current_generation: u32) -> u32 {
    if !bukkitCanContainBlock(block_number) {
        return GRID_BLOCK_INDEX_EMPTY;
    }

    let bukkit_index = calculateBukkitIndex(block_number);
    let bukkit_generation = atomicLoad(&sparse_grid.bukkit_generations[bukkit_index]);
    if bukkit_generation != current_generation {
        return GRID_BLOCK_INDEX_EMPTY;
    }

    let block_index = atomicLoad(&sparse_grid.block_index_bukkits[bukkit_index]);
    if block_index >= N_MAX_ACTIVE_BLOCKS {
        return GRID_BLOCK_INDEX_EMPTY;
    }

    return block_index;
}

fn retrieveBlockIndexFromBukkit(block_number: vec3i) -> u32 {
    return retrieveBlockIndexFromBukkitGeneration(block_number, sparse_grid.current_generation);
}

fn calculateCellIndexFromCellNumber(cell_number: vec3i) -> u32 {
    let block_number = calculateBlockNumberContainingCell(cell_number);
    let block_index = retrieveBlockIndexFromBukkit(block_number);

    // failsafe if something went wrong with allocation
    if block_index == GRID_BLOCK_INDEX_EMPTY {
        return GRID_BLOCK_INDEX_EMPTY;
    }

    let cell_index_within_block = calculateCellIndexWithinBlock(cell_number);
    return (block_index << LOG_BLOCK_SIZE_CUBED) + cell_index_within_block;
}

fn loadThreeCellStencilBlockNeighborhood(start_cell_number: vec3i) -> BlockNeighborhood {
    let start_block = calculateBlockNumberContainingCell(start_cell_number);
    let cell_offset = start_cell_number & vec3i(i32(BLOCK_MASK));
    let lower_block_delta = vec3i(
        select(0i, 1i, cell_offset.x == 0i),
        select(0i, 1i, cell_offset.y == 0i),
        select(0i, 1i, cell_offset.z == 0i),
    );
    let upper_block_delta = vec3i(
        select(0i, 1i, cell_offset.x == i32(BLOCK_MASK)),
        select(0i, 1i, cell_offset.y == i32(BLOCK_MASK)),
        select(0i, 1i, cell_offset.z == i32(BLOCK_MASK)),
    );

    var neighborhood: BlockNeighborhood;
    neighborhood.single_block_cell_index_base = GRID_BLOCK_INDEX_EMPTY;
    let last_grid_cell = gridLastCell();
    neighborhood.check_cell_range = select(
        0u,
        1u,
        any(start_cell_number <= vec3i(0)) || any(start_cell_number >= last_grid_cell)
    );

    let current_generation = sparse_grid.current_generation;
    if all(lower_block_delta == vec3i(0)) && all(upper_block_delta == vec3i(0)) {
        let block_index = retrieveBlockIndexFromBukkitGeneration(start_block, current_generation);
        neighborhood.min_block = start_block;
        neighborhood.block_indices[0] = block_index;
        if block_index != GRID_BLOCK_INDEX_EMPTY {
            neighborhood.single_block_cell_index_base = block_index << LOG_BLOCK_SIZE_CUBED;
        }
        return neighborhood;
    }

    let min_block = start_block - lower_block_delta;
    let max_block = start_block + upper_block_delta;
    let dims = vec3u(max_block - min_block + vec3i(1));

    neighborhood.min_block = min_block;

    for (var block_z = 0u; block_z < dims.z; block_z++) {
        for (var block_y = 0u; block_y < dims.y; block_y++) {
            for (var block_x = 0u; block_x < dims.x; block_x++) {
                let block_offset = vec3u(block_x, block_y, block_z);
                let cache_index = block_x + 2u * (block_y + 2u * block_z);
                neighborhood.block_indices[cache_index] = retrieveBlockIndexFromBukkitGeneration(
                    min_block + vec3i(block_offset),
                    current_generation,
                );
            }
        }
    }

    return neighborhood;
}

fn calculateCellIndexFromLoadedBlockNeighborhood(
    cell_number: vec3i,
    neighborhood: ptr<function, BlockNeighborhood>,
) -> u32 {
    let cell_index_within_block = calculateCellIndexWithinBlock(cell_number);
    return calculateCellIndexFromLoadedBlockNeighborhoodWithLocalIndex(
        cell_number,
        cell_index_within_block,
        neighborhood,
    );
}

fn calculateCellIndexFromLoadedBlockNeighborhoodWithLocalIndex(
    cell_number: vec3i,
    cell_index_within_block: u32,
    neighborhood: ptr<function, BlockNeighborhood>,
) -> u32 {
    if (*neighborhood).check_cell_range != 0u && !cellNumberInSparseGridRange(cell_number) {
        return GRID_BLOCK_INDEX_EMPTY;
    }

    let single_block_cell_index_base = (*neighborhood).single_block_cell_index_base;
    if single_block_cell_index_base != GRID_BLOCK_INDEX_EMPTY {
        return single_block_cell_index_base + cell_index_within_block;
    }

    let block_number = calculateBlockNumberContainingCell(cell_number);
    let block_offset = block_number - (*neighborhood).min_block;

    // Callers use cells bounded by the same min/max range used to load the
    // neighborhood, so the block offset is already in range here.
    let block_offset_u = vec3u(block_offset);
    let cache_index = block_offset_u.x + 2u * (block_offset_u.y + 2u * block_offset_u.z);
    let block_index = (*neighborhood).block_indices[cache_index];
    if block_index == GRID_BLOCK_INDEX_EMPTY {
        return GRID_BLOCK_INDEX_EMPTY;
    }

    return (block_index << LOG_BLOCK_SIZE_CUBED) + cell_index_within_block;
}
