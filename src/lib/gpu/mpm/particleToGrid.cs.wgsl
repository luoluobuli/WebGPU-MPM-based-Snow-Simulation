@group(1) @binding(0) var<storage, read_write> hash_map_entries: array<HashMapEntry>;
@group(1) @binding(2) var<storage, read_write> mapped_block_indexes: array<u32>;
@group(1) @binding(3) var<storage, read_write> grid_mass: array<atomic<i32>>;
@group(1) @binding(4) var<storage, read_write> grid_momentum_x: array<atomic<i32>>;
@group(1) @binding(5) var<storage, read_write> grid_momentum_y: array<atomic<i32>>;
@group(1) @binding(6) var<storage, read_write> grid_momentum_z: array<atomic<i32>>;

@group(2) @binding(0) var<storage, read_write> particleDataIn: array<ParticleData>;

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

    var shearResistance = SHEAR_RESISTANCE; // μ
    var volumetricResistance = VOLUME_RESISTANCE; // λ
    hardenLameParameters(particle.deformationPlastic, &shearResistance, &volumetricResistance);
    let stress = calculateStressFirstPiolaKirchhoff(particle.deformationElastic, shearResistance, volumetricResistance); // P
    let stressTranspose = transpose(stress);

    const DENSITY_KG_PER_M3 = 400.;
    const INVERSE_DENSITY = 1 / DENSITY_KG_PER_M3;
    let particleVolume = particle.mass * INVERSE_DENSITY; // V

    // enumerate the 3x3 neighborhood of cells around the cell that contains the particle
    var total_particle_stress_force = vec3f(0);
    for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
        for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
            for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                let cell_number = startCellNumber + vec3i(offsetX, offsetY, offsetZ);
                if !cellNumberInGridRange(cell_number) { continue; }

                let block_number = calculateBlockNumberContainingCell(cell_number);
                let block_index = retrieveBlockIndexFromHashMap(block_number);
                
                // failsafe if something went wrong with allocation
                // if block_ptr == GRID_HASH_MAP_BLOCK_INDEX_EMPTY { continue; }
                
                let cell_index_within_block = calculateCellIndexWithinBlock(cell_number);
                let cell_index = block_index * 64u + cell_index_within_block;
                
                // w
                let cellWeight = cellWeights[u32(offsetX + 1)].x
                    * cellWeights[u32(offsetY + 1)].y
                    * cellWeights[u32(offsetZ + 1)].z;

                // ∇w (gradient wrt fractional pos)
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
                
                let momentum = cellWeight * particleCurrentMomentum + stressMomentum;

                atomicAdd(&grid_momentum_x[cell_index], i32(momentum.x * uniforms.fixedPointScale));
                atomicAdd(&grid_momentum_y[cell_index], i32(momentum.y * uniforms.fixedPointScale));
                atomicAdd(&grid_momentum_z[cell_index], i32(momentum.z * uniforms.fixedPointScale));
                atomicAdd(&grid_mass[cell_index], i32(cellWeight * particle.mass * uniforms.fixedPointScale));
            }
        }
    }
}