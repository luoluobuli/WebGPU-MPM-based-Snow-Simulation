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

    let cellDims = calculateCellDims();
    let startCellNumber = calculateCellNumber(particle.pos, cellDims);
    let cellFracPos = calculateFractionalPosFromCellMin(particle.pos, cellDims, startCellNumber);
    let cellWeights = calculateQuadraticBSplineCellWeights(cellFracPos);
    let cellWeightsDeriv = calculateQuadraticBSplineCellWeightDerivatives(cellFracPos);


    const YOUNGS_MODULUS_PA = 1.4e5;
    const POISSONS_RATIO = 0.2;
    
    // Lamé parameters
    let shearModulus = YOUNGS_MODULUS_PA / (2 * (1 + POISSONS_RATIO));
    let bulkModulus = YOUNGS_MODULUS_PA * POISSONS_RATIO / ((1 + POISSONS_RATIO) * (1 - 2 * POISSONS_RATIO));
    let stress = calculateStressFirstPiolaKirchhoff(particle.deformationElastic, shearModulus, bulkModulus); // P
    let stressTranspose = transpose(stress);

    const DENSITY_KG_PER_M3 = 400.;
    const INVERSE_DENSITY = 1 / DENSITY_KG_PER_M3;
    let particleVolume = particle.mass * INVERSE_DENSITY; // V


    // enumerate the 3x3 neighborhood of cells around the cell that contains the particle
    var newParticleVelocity = vec3f(0); // cumulatively keep track of the cell's velocities, weighted using our kernel above
    for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
        for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
            for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                let cellNumber = startCellNumber + vec3i(offsetX, offsetY, offsetZ);
                if any(vec3i(0) > cellNumber) || any(cellNumber >= vec3i(i32(uniforms.gridResolution))) { continue; }



                let cellIndex = u32(cellNumber.x) + uniforms.gridResolution * (u32(cellNumber.y) + uniforms.gridResolution * u32(cellNumber.z));
                
                // w
                let cellWeight = cellWeights[u32(offsetX + 1)].x
                    * cellWeights[u32(offsetY + 1)].y
                    * cellWeights[u32(offsetZ + 1)].z;

                // ∇w (gradient wrt fractional pos)
                // divide by cell size to convert gradient to world space because fractional pos is dependent on cell size
                let cellWeightGradient = vec3f(
                    cellWeightsDeriv[u32(offsetX + 1)].x * cellWeights[u32(offsetY + 1)].y * cellWeights[u32(offsetZ + 1)].z,
                    cellWeights[u32(offsetX + 1)].x * cellWeightsDeriv[u32(offsetY + 1)].y * cellWeights[u32(offsetZ + 1)].z,
                    cellWeights[u32(offsetX + 1)].x * cellWeights[u32(offsetY + 1)].y * cellWeightsDeriv[u32(offsetZ + 1)].z
                ) / cellDims;
                

                // F = -V  Pᵀ  ∇w
                let stressForce = -particleVolume * stressTranspose * cellWeightGradient;

                // p = m v
                let particleCurrentMomentum = particle.mass * particle.vel;
                // dp = F dt
                let stressMomentum = stressForce * uniforms.simulationTimestep;
                

                // why is only the current momentum multiplied by cell weight?
                let momentum = cellWeight * particleCurrentMomentum + stressMomentum;


                atomicAdd(&gridDataOut[cellIndex].momentumX, i32(momentum.x * uniforms.fixedPointScale));
                atomicAdd(&gridDataOut[cellIndex].momentumY, i32(momentum.y * uniforms.fixedPointScale));
                atomicAdd(&gridDataOut[cellIndex].momentumZ, i32(momentum.z * uniforms.fixedPointScale));
                atomicAdd(&gridDataOut[cellIndex].mass, i32(cellWeight * particle.mass * uniforms.fixedPointScale));
            }
        }
    }
}