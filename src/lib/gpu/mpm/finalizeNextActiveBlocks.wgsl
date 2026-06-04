@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid: SparseGridStorage;

@group(3) @binding(0) var<storage, read_write> next_sparse_grid: SparseGridStorage;
@group(3) @binding(1) var<storage, read_write> next_active_block_dispatch_args: array<u32>;

const ACTIVE_BLOCK_DISPATCH_WIDTH = 256u;

@compute
@workgroup_size(1)
fn finalizeNextActiveBlockDispatch() {
    let count = min(atomicLoad(&next_sparse_grid.n_allocated_blocks), N_MAX_ACTIVE_BLOCKS);

    if count == 0u {
        next_active_block_dispatch_args[0] = 1u;
        next_active_block_dispatch_args[1] = 1u;
        next_active_block_dispatch_args[2] = 1u;
        next_active_block_dispatch_args[3] = 0u;
        return;
    }

    next_active_block_dispatch_args[0] = min(count, ACTIVE_BLOCK_DISPATCH_WIDTH);
    next_active_block_dispatch_args[1] = (count + ACTIVE_BLOCK_DISPATCH_WIDTH - 1u) / ACTIVE_BLOCK_DISPATCH_WIDTH;
    next_active_block_dispatch_args[2] = 1u;
    next_active_block_dispatch_args[3] = count;
}
