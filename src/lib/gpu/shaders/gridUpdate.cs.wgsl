@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(1) var<storage, read_write> gridData: array<GridData>;

@compute
@workgroup_size(256)
fn doGridUpdate(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let threadIndex = gid.x;
    if threadIndex >= arrayLength(&gridData) { return; }

    var grid = gridData[threadIndex];

    let mass = atomicLoad(&grid.mass);
    if (mass > 0.0) {
        var vel = vec3f(
            atomicLoad(&grid.vel.x),
            atomicLoad(&grid.vel.y),
            atomicLoad(&grid.vel.z)
        ) / mass;

        vel += vec3f(0.0, 0.0, -9.81) * uniforms.simulationTimestep;

        atomicStore(&grid.vel.x, vel.x);
        atomicStore(&grid.vel.y, vel.y);
        atomicStore(&grid.vel.z, vel.z);
    }
}