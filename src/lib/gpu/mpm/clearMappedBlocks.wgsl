@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(3) var<storage, read_write> grid_mass: array<i32>;
@group(1) @binding(4) var<storage, read_write> grid_momentum_x: array<i32>;
@group(1) @binding(5) var<storage, read_write> grid_momentum_y: array<i32>;
@group(1) @binding(6) var<storage, read_write> grid_momentum_z: array<i32>;

@compute
@workgroup_size(64) // 64 = # cells in a 4×4×4 block
fn clearMappedBlocks(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
) {
    let block_index = wid.y * 256u + wid.x; // TODO this calculation comes from magic numbers set in the runner
    let cell_index_in_block = lid.x;
    
    let cell_index = block_index * 64u + cell_index_in_block;
    if cell_index >= arrayLength(&grid_mass) { return; }
    
    grid_mass[cell_index] = 0;
    grid_momentum_x[cell_index] = 0;
    grid_momentum_y[cell_index] = 0;
    grid_momentum_z[cell_index] = 0;
}
