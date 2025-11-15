@group(1) @binding(1) var<storage, read_write> gridData: array<GridData>;

@compute
@workgroup_size(256)
fn doGridUpdate(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let threadIndex = gid.x;
    if threadIndex >= arrayLength(&gridData) { return; }

    let grid = &gridData[threadIndex];

    let mass = f32(atomicLoad(&(*grid).mass)) / uniforms.fixedPointScale;
    if (mass > 0.0) {
        var vel = vec3f(
            f32(atomicLoad(&(*grid).vx)) / uniforms.fixedPointScale,
            f32(atomicLoad(&(*grid).vy)) / uniforms.fixedPointScale,
            f32(atomicLoad(&(*grid).vz)) / uniforms.fixedPointScale,
        ) / mass;

        vel += vec3f(0.0, 0.0, -9.81) * uniforms.simulationTimestep;

        atomicStore(&(*grid).vx, i32(vel.x * uniforms.fixedPointScale));
        atomicStore(&(*grid).vy, i32(vel.y * uniforms.fixedPointScale));
        atomicStore(&(*grid).vz, i32(vel.z * uniforms.fixedPointScale));
    }
}