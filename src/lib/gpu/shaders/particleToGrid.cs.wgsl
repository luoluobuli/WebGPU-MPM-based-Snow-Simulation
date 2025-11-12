@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> particleDataIn: array<ParticleData>;
@group(1) @binding(1) var<storage, read_write> gridDataOut: array<GridData>;

@compute
@workgroup_size(256)
fn doParticleToGrid(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let threadIndex = gid.x;
    if threadIndex >= arrayLength(&particleDataIn) { return; }

    gridResolution = uniforms.gridResolution;
    dx = 1 / gridResolution;
    inv_dx = f32(gridResolution);

    let particle = particleDataIn[threadIndex];

    let grid_base = vec3i(particle.pos * inv_dx - 0.5);

    // fractional offset
    let fx = particle.pos * inv_dx - vec3f(grid_base);

    // quadratic kernel weights
    var w: array<vec3f, 3>;
    w[0] = 0.5 * (1.5 - fx) * (1.5 - fx);
    w[1] = 0.75 - (fx - 1.0) * (fx - 1.0);
    w[2] = 0.5 * (fx - 0.5) * (fx - 0.5);

    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            for (var k = 0u; k < 3u; k++) {
                let weight = w[i].x * w[j].y * w[k].z;
                let node = vec3u(grid_base) + vec3u(i, j, k);
                let nodeIndex = node.x + node.y * gridResolution + node.z * gridResolution * gridResolution;

                let contribVx = i32(weight * p.vel.x * uniforms.fpScale);
                let contribVy = i32(weight * p.vel.y * uniforms.fpScale);
                let contribVz = i32(weight * p.vel.z * uniforms.fpScale);
                let contribMass = i32(weight * p.mass * uniforms.fpScale);

                atomicAdd(&grid[nodeIndex].vx, contribVx);
                atomicAdd(&grid[nodeIndex].vy, contribVy);
                atomicAdd(&grid[nodeIndex].vz, contribVz);
                atomicAdd(&grid[nodeIndex].mass, contribMass);
            }
        }
    }
}