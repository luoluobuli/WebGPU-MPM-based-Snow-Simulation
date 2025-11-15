@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> particleDataOut: array<ParticleData>;
@group(1) @binding(1) var<storage, read_write> gridDataIn: array<GridData>;

@compute
@workgroup_size(256)
fn doGridToParticle(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let threadIndex = gid.x;
    if threadIndex >= arrayLength(&particleDataOut) { return; }

    let gridResolution = uniforms.gridResolution;
    let inv_dx = f32(gridResolution);

    var particle = particleDataOut[threadIndex];

    let grid_base = vec3i(particle.pos * inv_dx - 0.5);

    // fractional offset
    let fx = particle.pos * inv_dx - vec3f(grid_base);

    // quadratic kernel weights
    var w: array<vec3f, 3>;
    w[0] = 0.5 * (1.5 - fx) * (1.5 - fx);
    w[1] = 0.75 - (fx - 1.0) * (fx - 1.0);
    w[2] = 0.5 * (fx - 0.5) * (fx - 0.5);

    var new_vel = vec3f(0.0);

    // for (var i = 0u; i < 3u; i++) {
    //     for (var j = 0u; j < 3u; j++) {
    //         for (var k = 0u; k < 3u; k++) {
    //             let weight = w[i].x * w[j].y * w[k].z;
    //             let node = grid_base + vec3i(i32(i), i32(j), i32(k));
    //             if (any(vec3f(node) < uniforms.gridMinCoords) || any(vec3f(node) >= uniforms.gridMaxCoords)) {
    //                 continue;
    //             }
    //             let gridCellIndex = node.x + node.y * gridResolution + node.z * gridResolution * gridResolution;

    //             let gx = f32(atomicLoad(&gridDataIn[gridCellIndex].vx)) / uniforms.fixedPointScale;
    //             let gy = f32(atomicLoad(&gridDataIn[gridCellIndex].vy)) / uniforms.fixedPointScale;
    //             let gz = f32(atomicLoad(&gridDataIn[gridCellIndex].vz)) / uniforms.fixedPointScale;
    //             let grid_mass = f32(atomicLoad(&gridDataIn[gridCellIndex].mass)) / uniforms.fixedPointScale;

    //             if (grid_mass <= 0.0) { continue; }
    //             let grid_vel = vec3f(gx, gy, gz) / grid_mass;
    //             new_vel += weight * grid_vel;
    //         }
    //     }
    // }

    let gridCellSize = (uniforms.gridMaxCoords - uniforms.gridMinCoords) / f32(uniforms.gridResolution);
    let particlePosInGrid = particle.pos - uniforms.gridMinCoords;
    let gridStart = vec3u(
        u32(particlePosInGrid.x / gridCellSize.x),
        u32(particlePosInGrid.y / gridCellSize.y),
        u32(particlePosInGrid.z / gridCellSize.z),
    );

    // particle.pos = uniforms.gridMinCoords;

    if all(vec3u(0) <= gridStart) && all(gridStart < vec3u(uniforms.gridResolution)) {
        let gridCellIndex = gridStart.x + uniforms.gridResolution * (gridStart.y + uniforms.gridResolution * gridStart.z);

        new_vel = vec3f(
            0,
            0,
            f32(atomicLoad(&gridDataIn[gridCellIndex].vz)) / uniforms.fixedPointScale,
        );
    }


    particle.vel = new_vel;
    particle.pos += new_vel * uniforms.simulationTimestep;

    // particle.vel = vec3f(
    //     0,
    //     0,
    //     f32(atomicLoad(&gridDataIn[256].vz)) / uniforms.fixedPointScale
    // );
    // particle.vel += vec3f(0, 0, -9.81) * uniforms.simulationTimestep;
    // particle.pos += particle.vel * uniforms.simulationTimestep;

    particleDataOut[threadIndex] = particle;
}