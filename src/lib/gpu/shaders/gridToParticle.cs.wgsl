@group(1) @binding(0) var<storage, read_write> particleDataOut: array<ParticleData>;
@group(1) @binding(1) var<storage, read_write> gridDataIn: array<GridData>;

@compute
@workgroup_size(256)
fn doGridToParticle(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let threadIndex = gid.x;
    if threadIndex >= arrayLength(&particleDataOut) { return; }



    var particle = particleDataOut[threadIndex];



    // get the grid cell containing this particle
    let startCell = cellContainingPos(particle.pos);

    let fractionalPosFromCellMin = (particle.pos - startCell.minPos) / startCell.dims;
    let velocityWeightKernel = computeVelocityWeightsKernel(fractionalPosFromCellMin);



    // enumerate the 3x3 neighborhood of cells around the cell that contains the particle
    var newParticleVelocity = vec3f(0); // cumulatively keep track of the cell's velocities, weighted using our kernel above
    for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
        for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
            for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                let cellNumber = startCell.number + vec3i(offsetX, offsetY, offsetZ);

                if any(vec3i(0) > cellNumber) || any(cellNumber >= vec3i(uniforms.gridResolution)) { continue; }

                let cellIndex = u32(cellNumber.x) + uniforms.gridResolution * (u32(cellNumber.y) + uniforms.gridResolution * u32(cellNumber.z));
                
                let cellMass = f32(atomicLoad(&gridDataIn[cellIndex].mass)) / uniforms.fixedPointScale;
                if cellMass <= 0 { continue; }

                let cellMomentumX = f32(atomicLoad(&gridDataIn[cellIndex].vx)) / uniforms.fixedPointScale;
                let cellMomentumY = f32(atomicLoad(&gridDataIn[cellIndex].vy)) / uniforms.fixedPointScale;
                let cellMomentumZ = f32(atomicLoad(&gridDataIn[cellIndex].vz)) / uniforms.fixedPointScale;
                let cellVelocity = vec3f(cellMomentumX, cellMomentumY, cellMomentumZ) / cellMass;

                
                let cellWeight = velocityWeightKernel[u32(offsetX + 1)].x
                    * velocityWeightKernel[u32(offsetY + 1)].y
                    * velocityWeightKernel[u32(offsetZ + 1)].z;
                    
                newParticleVelocity += cellWeight * cellVelocity;
            }
        }
    }

    particle.vel = newParticleVelocity;
    particle.pos += newParticleVelocity * uniforms.simulationTimestep;


    particleDataOut[threadIndex] = particle;
}