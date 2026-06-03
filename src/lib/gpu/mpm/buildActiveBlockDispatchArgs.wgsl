@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid : SparseGridStorage;

@group(2) @binding(0) var<storage, read_write> active_block_dispatch_args: array<u32>;

@compute
@workgroup_size(1)
fn buildActiveBlockDispatchArgs() {
    let count = min(atomicLoad(&sparse_grid.n_allocated_blocks), N_MAX_ACTIVE_BLOCKS);
    let active_cell_count = count << LOG_BLOCK_SIZE_CUBED;
    let clear_cell_workgroups = (active_cell_count + 255u) / 256u;

    if count == 0u {
        active_block_dispatch_args[0] = 1u;
        active_block_dispatch_args[1] = 1u;
        active_block_dispatch_args[2] = 1u;
        active_block_dispatch_args[3] = 0u;
        active_block_dispatch_args[4] = 1u;
        active_block_dispatch_args[5] = 1u;
        active_block_dispatch_args[6] = 1u;
        active_block_dispatch_args[7] = 0u;
        return;
    }

    active_block_dispatch_args[0] = min(count, 256u);
    active_block_dispatch_args[1] = (count + 255u) / 256u;
    active_block_dispatch_args[2] = 1u;
    active_block_dispatch_args[3] = count;
    active_block_dispatch_args[4] = clear_cell_workgroups;
    active_block_dispatch_args[5] = 1u;
    active_block_dispatch_args[6] = 1u;
    active_block_dispatch_args[7] = active_cell_count;
}
