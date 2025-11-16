@group(1) @binding(0) var<storage, read_write> particleDataIn: array<ParticleData>;
@group(1) @binding(1) var<storage, read_write> gridDataOut: array<CellData>;

@compute
@workgroup_size(256)
fn doParticleToGrid(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let threadIndex = gid.x;
    if threadIndex >= arrayLength(&particleDataIn) { return; }



    let particle = particleDataIn[threadIndex];
    let particleInfo = calculateMpmParticleCellInfo(particle.pos);



    // enumerate the 3x3 neighborhood of cells around the cell that contains the particle
    var newParticleVelocity = vec3f(0); // cumulatively keep track of the cell's velocities, weighted using our kernel above
    for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
        for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
            for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                let cellNumber = particleInfo.startCellNumber + vec3i(offsetX, offsetY, offsetZ);
                if any(vec3i(0) > cellNumber) || any(cellNumber >= vec3i(i32(uniforms.gridResolution))) { continue; }



                let cellIndex = u32(cellNumber.x) + uniforms.gridResolution * (u32(cellNumber.y) + uniforms.gridResolution * u32(cellNumber.z));
                

                
                let cellWeight = particleInfo.velocityWeightsKernel[u32(offsetX + 1)].x
                    * particleInfo.velocityWeightsKernel[u32(offsetY + 1)].y
                    * particleInfo.velocityWeightsKernel[u32(offsetZ + 1)].z;

                let contribVx = cellWeight * particle.vel.x * particle.mass * uniforms.fixedPointScale;
                let contribVy = cellWeight * particle.vel.y * particle.mass * uniforms.fixedPointScale;
                let contribVz = cellWeight * particle.vel.z * particle.mass * uniforms.fixedPointScale;
                let contribMass = cellWeight * particle.mass * uniforms.fixedPointScale;

                atomicAdd(&gridDataOut[cellIndex].momentumX, i32(contribVx));
                atomicAdd(&gridDataOut[cellIndex].momentumY, i32(contribVy));
                atomicAdd(&gridDataOut[cellIndex].momentumZ, i32(contribVz));
                atomicAdd(&gridDataOut[cellIndex].mass, i32(contribMass));
            }
        }
    }
}