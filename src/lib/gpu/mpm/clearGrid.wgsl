// clearGrid.wgsl - Clear dense grid buffers

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(3) var<storage, read_write> grid_mass: array<i32>;
@group(1) @binding(4) var<storage, read_write> grid_momentum_x: array<i32>;
@group(1) @binding(5) var<storage, read_write> grid_momentum_y: array<i32>;
@group(1) @binding(6) var<storage, read_write> grid_momentum_z: array<i32>;

@compute @workgroup_size(256)
fn clearGrid(@builtin(global_invocation_id) gid: vec3u) {
    let totalCells = uniforms.gridResolution.x * uniforms.gridResolution.y * uniforms.gridResolution.z;
    let cellIndex = gid.x;
    
    if (cellIndex >= totalCells) { return; }
    
    grid_mass[cellIndex] = 0;
    grid_momentum_x[cellIndex] = 0;
    grid_momentum_y[cellIndex] = 0;
    grid_momentum_z[cellIndex] = 0;
}
