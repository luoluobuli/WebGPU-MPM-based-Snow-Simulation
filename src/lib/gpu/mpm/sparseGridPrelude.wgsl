const BLOCK_SIZE = 4u;
const BLOCK_SIZE_CUBED = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
const LOG_BLOCK_SIZE = 2u; // log2
const BLOCK_MASK = 3u; // 4 - 1

const BUKKIT_DOMAIN_GRID_RESOLUTION = 384u;
const BUKKIT_DOMAIN_BLOCKS_X = (BUKKIT_DOMAIN_GRID_RESOLUTION + BLOCK_SIZE - 1u) / BLOCK_SIZE;
const BUKKIT_DOMAIN_BLOCKS_Y = BUKKIT_DOMAIN_BLOCKS_X;
const BUKKIT_DOMAIN_BLOCKS_Z = BUKKIT_DOMAIN_BLOCKS_X;
const BUKKIT_DOMAIN_BLOCK_COUNT = BUKKIT_DOMAIN_BLOCKS_X * BUKKIT_DOMAIN_BLOCKS_Y * BUKKIT_DOMAIN_BLOCKS_Z;

const N_MAX_ACTIVE_BLOCKS = 100000u;
const GRID_BLOCK_INDEX_EMPTY = 0xFFFFFFFFu;
const GRID_BLOCK_INDEX_RESERVED = 0xFFFFFFFEu;

struct SparseGridStorage {
    n_allocated_blocks: atomic<u32>,
    _padding: array<u32, 3>,
    block_index_bukkits: array<atomic<u32>, BUKKIT_DOMAIN_BLOCK_COUNT>,
    mapped_block_numbers: array<vec4i, N_MAX_ACTIVE_BLOCKS>,
    block_particle_counts: array<atomic<u32>, N_MAX_ACTIVE_BLOCKS>,
    block_particle_offsets: array<atomic<u32>, N_MAX_ACTIVE_BLOCKS>,
}

fn calculateBlockNumberContainingCell(cell_number: vec3i) -> vec3i {
    return cell_number >> vec3u(LOG_BLOCK_SIZE);
}

fn calculateCellIndexWithinBlock(cell_number: vec3i) -> u32 {
    let cell_index_within_block = cell_number & vec3i(i32(BLOCK_MASK));
    return u32(cell_index_within_block.x + cell_index_within_block.y * 4 + cell_index_within_block.z * 16);
}

fn bukkitCanContainBlock(block_number: vec3i) -> bool {
    return all(block_number >= vec3i(0))
        && u32(block_number.x) < BUKKIT_DOMAIN_BLOCKS_X
        && u32(block_number.y) < BUKKIT_DOMAIN_BLOCKS_Y
        && u32(block_number.z) < BUKKIT_DOMAIN_BLOCKS_Z;
}

fn calculateBukkitIndex(block_number: vec3i) -> u32 {
    let block = vec3u(block_number);
    return block.x + BUKKIT_DOMAIN_BLOCKS_X * (block.y + BUKKIT_DOMAIN_BLOCKS_Y * block.z);
}

fn retrieveBlockIndexFromBukkit(block_number: vec3i) -> u32 {
    if !bukkitCanContainBlock(block_number) {
        return GRID_BLOCK_INDEX_EMPTY;
    }

    let bukkit_index = calculateBukkitIndex(block_number);
    let block_index = atomicLoad(&sparse_grid.block_index_bukkits[bukkit_index]);
    if block_index >= N_MAX_ACTIVE_BLOCKS {
        return GRID_BLOCK_INDEX_EMPTY;
    }

    return block_index;
}

fn calculateCellIndexFromCellNumber(cell_number: vec3i) -> u32 {
    let block_number = calculateBlockNumberContainingCell(cell_number);
    let block_index = retrieveBlockIndexFromBukkit(block_number);

    // failsafe if something went wrong with allocation
    if block_index == GRID_BLOCK_INDEX_EMPTY {
        return GRID_BLOCK_INDEX_EMPTY;
    }

    let cell_index_within_block = calculateCellIndexWithinBlock(cell_number);
    return block_index * 64u + cell_index_within_block;
}
