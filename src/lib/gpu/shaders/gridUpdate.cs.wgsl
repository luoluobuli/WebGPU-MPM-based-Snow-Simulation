@group(1) @binding(1) var<storage, read_write> gridData: array<GridData>;

@compute
@workgroup_size(256)
fn doGridUpdate(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let threadIndex = gid.x;
    if threadIndex >= arrayLength(&gridData) { return; }

    let grid = &gridData[threadIndex];

    let cellMass = f32(atomicLoad(&(*grid).mass)) / uniforms.fixedPointScale;
    if cellMass <= 0 { return; }


    let cellForce = vec3f(0.0, 0.0, -9.81) * cellMass;
    let cellMomentumChange = cellForce * uniforms.simulationTimestep;


    atomicAdd(&(*grid).vx, i32(cellMomentumChange.x * uniforms.fixedPointScale));
    atomicAdd(&(*grid).vy, i32(cellMomentumChange.y * uniforms.fixedPointScale));
    atomicAdd(&(*grid).vz, i32(cellMomentumChange.z * uniforms.fixedPointScale));
}