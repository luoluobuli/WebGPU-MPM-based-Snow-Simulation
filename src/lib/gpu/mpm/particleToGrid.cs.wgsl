@group(1) @binding(0) var<storage, read_write> hash_map_entries: array<HashMapEntry>;
@group(1) @binding(2) var<storage, read_write> mapped_block_indexes: array<u32>;
@group(1) @binding(3) var<storage, read_write> grid_mass: array<atomic<i32>>;
@group(1) @binding(4) var<storage, read_write> grid_momentum_x: array<atomic<i32>>;
@group(1) @binding(5) var<storage, read_write> grid_momentum_y: array<atomic<i32>>;
@group(1) @binding(6) var<storage, read_write> grid_momentum_z: array<atomic<i32>>;
@group(1) @binding(7) var<storage, read_write> grid_mass_displacement_x: array<atomic<i32>>;
@group(1) @binding(8) var<storage, read_write> grid_mass_displacement_y: array<atomic<i32>>;
@group(1) @binding(9) var<storage, read_write> grid_mass_displacement_z: array<atomic<i32>>;

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

    if uniforms.use_pbmpm == 0 {

        // enumerate the 3x3 neighborhood of cells around the cell that contains the particle
        for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
            for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
                for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                    let cell_number = startCellNumber + vec3i(offsetX, offsetY, offsetZ);
                    if !cellNumberInGridRange(cell_number) { continue; }

                    let cell_index = calculateCellIndexFromCellNumber(cell_number);
                    if cell_index == GRID_HASH_MAP_BLOCK_INDEX_EMPTY { continue; }
                    
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
                    
                    // a = -V  Pᵀ  ∇w
                    let stress_acceleration = -particleVolume * stressTranspose * cellWeightGradient;

                    // p = m v
                    let particleCurrentMomentum = particle.mass * particle.vel;
                    // dp = F dt
                    let stress_velocity = stress_acceleration * uniforms.simulationTimestep;
                    
                    let momentum = cellWeight * (particleCurrentMomentum + particle.mass * stress_velocity);

                    atomicAdd(&grid_momentum_x[cell_index], i32(momentum.x * uniforms.fixedPointScale));
                    atomicAdd(&grid_momentum_y[cell_index], i32(momentum.y * uniforms.fixedPointScale));
                    atomicAdd(&grid_momentum_z[cell_index], i32(momentum.z * uniforms.fixedPointScale));
                    atomicAdd(&grid_mass[cell_index], i32(cellWeight * particle.mass * uniforms.fixedPointScale));
                }
            }
        }
    }

    else {
        for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
            for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
                for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                    let cell_number = startCellNumber + vec3i(offsetX, offsetY, offsetZ);
                    if !cellNumberInGridRange(cell_number) { continue; }

                    let cell_index = calculateCellIndexFromCellNumber(cell_number);
                    if cell_index == GRID_HASH_MAP_BLOCK_INDEX_EMPTY { continue; }
                    
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
                    
                    // a = -V  Pᵀ  ∇w
                    let stress_acceleration = -particleVolume * stressTranspose * cellWeightGradient;

                    // p = m v
                    let particleCurrentMomentum = particle.mass * (particle.pos_displacement / uniforms.simulationTimestep);
                    // dp = F dt
                    let stress_velocity = stress_acceleration * uniforms.simulationTimestep;

                    let weighted_mass = cellWeight * particle.mass;

                    let cell_particle_offset = vec3f(cell_number) - (vec3f(startCellNumber) + cellFracPos) + 0.5;
                    let affine_displacement = particle.deformation_displacement * cell_particle_offset;

                    let momentum = weighted_mass * (particle.pos_displacement + affine_displacement) / uniforms.simulationTimestep;


                    atomicAdd(&grid_momentum_x[cell_index], i32(momentum.x * uniforms.fixedPointScale));
                    atomicAdd(&grid_momentum_y[cell_index], i32(momentum.y * uniforms.fixedPointScale));
                    atomicAdd(&grid_momentum_z[cell_index], i32(momentum.z * uniforms.fixedPointScale));
                    atomicAdd(&grid_mass[cell_index], i32(weighted_mass * uniforms.fixedPointScale));

                    // let cell_number = startCellNumber + vec3i(offsetX, offsetY, offsetZ);
                    // if !cellNumberInGridRange(cell_number) { continue; }

                    // let cell_index = calculateCellIndexFromCellNumber(cell_number);
                    // if cell_index == GRID_HASH_MAP_BLOCK_INDEX_EMPTY { continue; }
                    
                    // // w
                    // let cellWeight = cellWeights[u32(offsetX + 1)].x
                    //     * cellWeights[u32(offsetY + 1)].y
                    //     * cellWeights[u32(offsetZ + 1)].z;

                    // // ∇w (gradient wrt fractional pos)
                    // let cellWeightGradient = vec3f(
                    //     cellWeightsDeriv[u32(offsetX + 1)].x * cellWeights[u32(offsetY + 1)].y * cellWeights[u32(offsetZ + 1)].z,
                    //     cellWeights[u32(offsetX + 1)].x * cellWeightsDeriv[u32(offsetY + 1)].y * cellWeights[u32(offsetZ + 1)].z,
                    //     cellWeights[u32(offsetX + 1)].x * cellWeights[u32(offsetY + 1)].y * cellWeightsDeriv[u32(offsetZ + 1)].z
                    // ) / cellDims;
                    
                    // // // F = -V  Pᵀ  ∇w
                    // // let stressForce = -particleVolume * stressTranspose * cellWeightGradient;

                    // // // p = m v
                    // let cell_center_pos = uniforms.gridMinCoords + cellDims * (vec3f(cell_number) + vec3f(0.5));
                    // let particle_current_mass_displacement = particle.mass * (particle.pos_displacement + particle.deformation_displacement * (particle.pos - cell_center_pos));
                    // // // dp = F dt
                    // // let stressMomentum = stressForce * uniforms.simulationTimestep * uniforms.simulationTimestep;
                    
                    // let mass_displacement = cellWeight * particle_current_mass_displacement/* + stressMomentum*/;

                    // atomicAdd(&grid_mass_displacement_x[cell_index], i32(mass_displacement.x * uniforms.fixedPointScale));
                    // atomicAdd(&grid_mass_displacement_y[cell_index], i32(mass_displacement.y * uniforms.fixedPointScale));
                    // atomicAdd(&grid_mass_displacement_z[cell_index], i32(mass_displacement.z * uniforms.fixedPointScale));
                    // atomicAdd(&grid_mass[cell_index], i32(cellWeight * particle.mass * uniforms.fixedPointScale));
                }
            }
        }
    }
}