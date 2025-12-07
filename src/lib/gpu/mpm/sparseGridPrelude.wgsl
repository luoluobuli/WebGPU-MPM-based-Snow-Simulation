const BLOCK_SIZE = 4u;
const BLOCK_SIZE_CUBED = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
const LOG_BLOCK_SIZE = 2u; // log2
const BLOCK_MASK = 3u; // 4 - 1

const HASH_MAP_SIZE = 200003u; // prime modulus that is greater than N_MAX_BLOCKS_IN_HASH_MAP / load_limit (here 0.5)
const N_MAX_BLOCKS_IN_HASH_MAP = 100000u;
const N_HASH_MAP_CANDIDATE_INDEX_ATTEMPTS = 100u;
const GRID_HASH_MAP_BLOCK_INDEX_EMPTY = 0xFFFFFFFFu;
const GRID_HASH_MAP_BLOCK_INDEX_RESERVED = 0xFFFFFFFEu;

struct HashMapEntry {
    block_number: vec3i,
    block_index: atomic<u32>,
}

struct SparseGridStorage {
    n_allocated_blocks: atomic<u32>,
    // implicit 12 byte padding
    hash_map_entries: array<HashMapEntry, HASH_MAP_SIZE>,
    mapped_block_indexes: array<u32, N_MAX_BLOCKS_IN_HASH_MAP>,
    block_particle_counts: array<atomic<u32>, N_MAX_BLOCKS_IN_HASH_MAP>,
    block_particle_offsets: array<atomic<u32>, N_MAX_BLOCKS_IN_HASH_MAP>,
}

fn calculateBlockNumberContainingCell(cell_number: vec3i) -> vec3i {
    return cell_number >> vec3u(LOG_BLOCK_SIZE);
}

fn calculateCellIndexWithinBlock(cell_number: vec3i) -> u32 {
    let cell_index_within_block = cell_number & vec3i(i32(BLOCK_MASK));
    return u32(cell_index_within_block.x + cell_index_within_block.y * 4 + cell_index_within_block.z * 16);
}

fn retrieveBlockIndexFromHashMap(block_coord: vec3<i32>) -> u32 {
    let hash_key = hash3(bitcast<vec3u>(block_coord));

    for (var i = 0u; i < N_HASH_MAP_CANDIDATE_INDEX_ATTEMPTS; i++) {
        let candidate_index = (hash_key + i) % HASH_MAP_SIZE;
        let stored_coord = sparse_grid.hash_map_entries[candidate_index].block_number;
        let block_index = atomicLoad(&sparse_grid.hash_map_entries[candidate_index].block_index);
        
        if block_index == GRID_HASH_MAP_BLOCK_INDEX_EMPTY {
            return GRID_HASH_MAP_BLOCK_INDEX_EMPTY;
        }
        
        if all(stored_coord == block_coord) {
            return block_index;
        }
    }
    return GRID_HASH_MAP_BLOCK_INDEX_EMPTY;
}

fn calculateCellIndexFromCellNumber(cell_number: vec3i) -> u32 {
    let block_number = calculateBlockNumberContainingCell(cell_number);
    let block_index = retrieveBlockIndexFromHashMap(block_number);

    // failsafe if something went wrong with allocation
    if block_index == GRID_HASH_MAP_BLOCK_INDEX_EMPTY {
        return GRID_HASH_MAP_BLOCK_INDEX_EMPTY;
    }

    let cell_index_within_block = calculateCellIndexWithinBlock(cell_number);
    return block_index * 64u + cell_index_within_block;
}