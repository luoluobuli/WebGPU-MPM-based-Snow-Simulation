@group(1) @binding(1) var<storage, read_write> grid_momentum_x: array<atomic<i32>>;
@group(1) @binding(2) var<storage, read_write> grid_momentum_y: array<atomic<i32>>;
@group(1) @binding(3) var<storage, read_write> grid_momentum_z: array<atomic<i32>>;
@group(1) @binding(4) var<storage, read_write> grid_mass: array<atomic<i32>>;

@compute
@workgroup_size(8, 8, 4)
fn doClearGrid(
    @builtin(global_invocation_id) gid: vec3u,
) {
    if gid.x >= uniforms.gridResolution.x || gid.y >= uniforms.gridResolution.y || gid.z >= uniforms.gridResolution.z {
        return;
    }

    let thread_index = gid.x + uniforms.gridResolution.x * (gid.y + uniforms.gridResolution.y * gid.z);
    if thread_index >= arrayLength(&grid_mass) { return; }

    atomicStore(&grid_momentum_x[thread_index], 0);
    atomicStore(&grid_momentum_y[thread_index], 0);
    atomicStore(&grid_momentum_z[thread_index], 0);
    atomicStore(&grid_mass[thread_index], 0);
}
