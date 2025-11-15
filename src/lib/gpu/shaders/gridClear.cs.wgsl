@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(1) var<storage, read_write> gridData: array<GridData>;

@compute
@workgroup_size(256)
fn doClearGrid(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let threadIndex = gid.x;
    if threadIndex >= arrayLength(&gridData) { return; }

    let grid = &gridData[threadIndex];

    atomicStore(&(*grid).vx, 0);
    atomicStore(&(*grid).vy, 0);
    atomicStore(&(*grid).vz, 0);
    atomicStore(&(*grid).mass, 0);
}
