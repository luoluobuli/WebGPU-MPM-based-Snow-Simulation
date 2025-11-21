@group(1) @binding(0) var<storage, read_write> particleDataOut: array<ParticleData>;
@group(1) @binding(1) var<storage, read_write> gridDataIn: array<CellData>;

@compute
@workgroup_size(256)
fn doGridToParticle(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let threadIndex = gid.x;
    if threadIndex >= arrayLength(&particleDataOut) { return; }



    var particle = particleDataOut[threadIndex];

    let cellDims = calculateCellDims();
    let startCellNumber = calculateCellNumber(particle.pos, cellDims);
    let cellFracPos = calculateFractionalPosFromCellMin(particle.pos, cellDims, startCellNumber);
    let cellWeights = calculateQuadraticBSplineCellWeights(cellFracPos);
    let cellWeightsDeriv = calculateQuadraticBSplineCellWeightDerivatives(cellFracPos);



    // enumerate the 3x3 neighborhood of cells around the cell that contains the particle
    var newParticleVelocity = vec3f(0); // cumulatively keep track of the cell's velocities, weighted using our kernel above
    var totalVelocityGradient = mat3x3f();
    for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
        for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
            for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                let cellNumber = startCellNumber + vec3i(offsetX, offsetY, offsetZ);
                if any(vec3i(0) > cellNumber) || any(cellNumber >= vec3i(i32(uniforms.gridResolution))) { continue; }



                let cellIndex = u32(cellNumber.x) + uniforms.gridResolution * (u32(cellNumber.y) + uniforms.gridResolution * u32(cellNumber.z));
                
                let cellMass = f32(atomicLoad(&gridDataIn[cellIndex].mass)) / uniforms.fixedPointScale;
                if cellMass <= 0 { continue; }



                let cellMomentumX = f32(atomicLoad(&gridDataIn[cellIndex].momentumX)) / uniforms.fixedPointScale;
                let cellMomentumY = f32(atomicLoad(&gridDataIn[cellIndex].momentumY)) / uniforms.fixedPointScale;
                let cellMomentumZ = f32(atomicLoad(&gridDataIn[cellIndex].momentumZ)) / uniforms.fixedPointScale;
                let cellVelocity = vec3f(cellMomentumX, cellMomentumY, cellMomentumZ) / cellMass;

                
                let cellWeight = cellWeights[u32(offsetX + 1)].x
                    * cellWeights[u32(offsetY + 1)].y
                    * cellWeights[u32(offsetZ + 1)].z;
                    
                newParticleVelocity += cellWeight * cellVelocity;



                let cellWeightGradient = vec3f(
                    cellWeightsDeriv[u32(offsetX + 1)].x * cellWeights[u32(offsetY + 1)].y * cellWeights[u32(offsetZ + 1)].z,
                    cellWeights[u32(offsetX + 1)].x * cellWeightsDeriv[u32(offsetY + 1)].y * cellWeights[u32(offsetZ + 1)].z,
                    cellWeights[u32(offsetX + 1)].x * cellWeights[u32(offsetY + 1)].y * cellWeightsDeriv[u32(offsetZ + 1)].z,
                );

                totalVelocityGradient += mat3x3f(
                    cellWeightGradient.x * cellVelocity,
                    cellWeightGradient.y * cellVelocity,
                    cellWeightGradient.z * cellVelocity,
                );
            }
        }
    }



    particle.vel = newParticleVelocity;
    particle.pos += newParticleVelocity * uniforms.simulationTimestep;
    particle.deformationElastic += totalVelocityGradient * uniforms.simulationTimestep;

    
    applyPlasticity(&particle.deformationElastic, &particle.deformationPlastic);
    



    if particle.pos.x < uniforms.gridMinCoords.x {
        particle.vel.x *= -0.5;
        particle.pos.x = uniforms.gridMinCoords.x;
    }
    if particle.pos.x >= uniforms.gridMaxCoords.x {
        particle.vel.x *= -0.5;
        particle.pos.x = uniforms.gridMaxCoords.x;
    }

    if particle.pos.y < uniforms.gridMinCoords.y {
        particle.vel.y *= -0.5;
        particle.pos.y = uniforms.gridMinCoords.y;
    }
    if particle.pos.y >= uniforms.gridMaxCoords.y {
        particle.vel.y *= -0.5;
        particle.pos.y = uniforms.gridMaxCoords.y;
    }

    if particle.pos.z < uniforms.gridMinCoords.z {
        particle.vel.z *= -0.5;
        particle.pos.z = uniforms.gridMinCoords.z;
    }
    if particle.pos.z >= uniforms.gridMaxCoords.z {
        particle.vel.z *= -0.5;
        particle.pos.z = uniforms.gridMaxCoords.z;
    }




    particleDataOut[threadIndex] = particle;
}