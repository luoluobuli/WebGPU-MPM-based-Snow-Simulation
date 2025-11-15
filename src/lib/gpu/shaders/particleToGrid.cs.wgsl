@group(1) @binding(0) var<storage, read_write> particleDataIn: array<ParticleData>;
@group(1) @binding(1) var<storage, read_write> gridDataOut: array<GridData>;

@compute
@workgroup_size(256)
fn doParticleToGrid(
    @builtin(global_invocation_id) gid: vec3u,
) {
    // let threadIndex = gid.x;
    // if threadIndex >= arrayLength(&particleDataIn) { return; }

    // let gridResolution = uniforms.gridResolution;
    // let inv_dx = f32(gridResolution);
    // let dx = 1 / inv_dx;

    // let particle = particleDataIn[threadIndex];

    // let grid_base = vec3i(particle.pos * inv_dx - 0.5);

    // // fractional offset
    // let fx = particle.pos * inv_dx - vec3f(grid_base);

    // // quadratic kernel weights
    // var w: array<vec3f, 3>;
    // w[0] = 0.5 * (1.5 - fx) * (1.5 - fx);
    // w[1] = 0.75 - (fx - 1.0) * (fx - 1.0);
    // w[2] = 0.5 * (fx - 0.5) * (fx - 0.5);

    // for (var i = 0u; i < 3u; i++) {
    //     for (var j = 0u; j < 3u; j++) {
    //         for (var k = 0u; k < 3u; k++) {
    //             let weight = w[i].x * w[j].y * w[k].z;
    //             let node = grid_base + vec3i(i32(i), i32(j), i32(k));
    //             let nodeIndex = node.x + node.y * gridResolution + node.z * gridResolution * gridResolution;

    //             let contribVx = weight * particle.vel.x * uniforms.fixedPointScale;
    //             let contribVy = weight * particle.vel.y * uniforms.fixedPointScale;
    //             let contribVz = weight * particle.vel.z * uniforms.fixedPointScale;
    //             let contribMass = weight * particle.mass * uniforms.fixedPointScale;

    //             atomicAdd(&gridDataOut[nodeIndex].vx, i32(contribVx));
    //             atomicAdd(&gridDataOut[nodeIndex].vy, i32(contribVy));
    //             atomicAdd(&gridDataOut[nodeIndex].vz, i32(contribVz));
    //             atomicAdd(&gridDataOut[nodeIndex].mass, i32(contribMass));
    //         }
    //     }
    // }
}