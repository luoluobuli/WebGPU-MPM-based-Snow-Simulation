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

    let gridCellSize = (uniforms.gridMaxCoords - uniforms.gridMinCoords) / f32(uniforms.gridResolution);
    let particlePosInGrid = particle.pos - uniforms.gridMinCoords;
    let gridStart = vec3i(
        i32(particlePosInGrid.x / gridCellSize.x),
        i32(particlePosInGrid.y / gridCellSize.y),
        i32(particlePosInGrid.z / gridCellSize.z),
    );

    for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
        for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
            for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                let gridCell = gridStart + vec3i(offsetX, offsetY, offsetZ);

                if any(vec3i(0) > gridCell) || any(gridCell >= vec3i(uniforms.gridResolution)) { continue; }

                let weight = w[u32(offsetX + 1)].x * w[u32(offsetY + 1)].y * w[u32(offsetZ + 1)].z;
                let gridCellIndex = u32(gridStart.x) + uniforms.gridResolution * (u32(gridStart.y) + uniforms.gridResolution * u32(gridStart.z));
                
                let gx = f32(atomicLoad(&gridDataIn[gridCellIndex].vx)) / uniforms.fixedPointScale;
                let gy = f32(atomicLoad(&gridDataIn[gridCellIndex].vy)) / uniforms.fixedPointScale;
                let gz = f32(atomicLoad(&gridDataIn[gridCellIndex].vz)) / uniforms.fixedPointScale;
                let grid_mass = f32(atomicLoad(&gridDataIn[gridCellIndex].mass)) / uniforms.fixedPointScale;

                if (grid_mass <= 0.0) { continue; }
                let grid_vel = vec3f(gx, gy, gz) / grid_mass;
                new_vel += weight * grid_vel;
            }
        }
    }


    particle.vel = new_vel;
    particle.pos += new_vel * uniforms.simulationTimestep;

    particleDataOut[threadIndex] = particle;
}