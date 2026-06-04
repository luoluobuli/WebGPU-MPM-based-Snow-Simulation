@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid: SparseGridStorage;

struct GridAccumulatorAtomic {
    mass: atomic<i32>,
    momentum_x: atomic<i32>,
    momentum_y: atomic<i32>,
    momentum_z: atomic<i32>,
}

@group(1) @binding(3) var<storage, read_write> grid_accumulator: array<GridAccumulatorAtomic>;

const ACTIVE_BLOCK_DISPATCH_WIDTH = 256u;

@compute
@workgroup_size(64)
fn clearGridAccumulator(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
) {
    let active_block_index = wid.y * ACTIVE_BLOCK_DISPATCH_WIDTH + wid.x;
    let active_block_count = min(atomicLoad(&sparse_grid.n_allocated_blocks), N_MAX_ACTIVE_BLOCKS);
    if active_block_index >= active_block_count { return; }

    let cell_index = (active_block_index << LOG_BLOCK_SIZE_CUBED) + lid.x;
    atomicStore(&grid_accumulator[cell_index].mass, 0i);
    atomicStore(&grid_accumulator[cell_index].momentum_x, 0i);
    atomicStore(&grid_accumulator[cell_index].momentum_y, 0i);
    atomicStore(&grid_accumulator[cell_index].momentum_z, 0i);
}
