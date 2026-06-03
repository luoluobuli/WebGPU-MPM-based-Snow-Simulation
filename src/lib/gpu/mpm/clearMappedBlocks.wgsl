@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid: SparseGridStorage;
@group(1) @binding(3) var<storage, read_write> grid_mass: array<i32>;
@group(1) @binding(4) var<storage, read_write> grid_momentum_x: array<i32>;
@group(1) @binding(5) var<storage, read_write> grid_momentum_y: array<i32>;
@group(1) @binding(6) var<storage, read_write> grid_momentum_z: array<i32>;

@group(2) @binding(0) var<storage, read> active_block_dispatch_args: array<u32>;

@compute
@workgroup_size(256)
fn clearMappedBlocks(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
) {
    let cell_index = gid.x;
    if wid.x + 1u == num_workgroups.x {
        let active_cell_count = active_block_dispatch_args[7];
        if cell_index >= active_cell_count { return; }
    }

    grid_mass[cell_index] = 0;
    grid_momentum_x[cell_index] = 0;
    grid_momentum_y[cell_index] = 0;
    grid_momentum_z[cell_index] = 0;
}
