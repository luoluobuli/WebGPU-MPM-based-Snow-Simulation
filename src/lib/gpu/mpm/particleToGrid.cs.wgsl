@group(1) @binding(0) var<storage, read_write> hash_map_entries: array<HashMapEntry>;
@group(1) @binding(2) var<storage, read_write> mapped_block_indexes: array<u32>;
@group(1) @binding(3) var<storage, read_write> grid_mass: array<atomic<i32>>;
@group(1) @binding(4) var<storage, read_write> grid_momentum_x: array<atomic<i32>>;
@group(1) @binding(5) var<storage, read_write> grid_momentum_y: array<atomic<i32>>;
@group(1) @binding(6) var<storage, read_write> grid_momentum_z: array<atomic<i32>>;

@group(2) @binding(0) var<storage, read_write> particleDataIn: array<ParticleData>;
@group(2) @binding(1) var<storage, read_write> sortedParticleIndices: array<u32>;

@compute
@workgroup_size(256)
fn doParticleToGrid(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let thread_index = gid.x;
    if thread_index >= arrayLength(&particleDataIn) { return; }

    let particle_index = sortedParticleIndices[thread_index];
    let particle = particleDataIn[particle_index];

    let cell_dims = calculateCellDims();
    let start_cell_number = calculateCellNumber(particle.pos, cell_dims);
    let cell_frac_pos = calculateFractionalPosFromCellMin(particle.pos, cell_dims, start_cell_number);
    let cell_weights = calculateQuadraticBSplineCellWeights(cell_frac_pos);

    if uniforms.use_pbmpm == 0 {
        let cell_weights_deriv = calculateQuadraticBSplineCellWeightDerivatives(cell_frac_pos);

        var shear_resistance = SHEAR_RESISTANCE; // μ
        var volumetric_resistance = VOLUME_RESISTANCE; // λ
        hardenLameParameters(particle.deformationPlastic, &shear_resistance, &volumetric_resistance);
        let stress = calculateStressFirstPiolaKirchhoff(particle.deformationElastic, shear_resistance, volumetric_resistance); // P
        let stressTranspose = transpose(stress);

        const DENSITY_KG_PER_M3 = 400.;
        const INVERSE_DENSITY = 1 / DENSITY_KG_PER_M3;
        let particleVolume = particle.mass * INVERSE_DENSITY; // V

        // enumerate the 3x3 neighborhood of cells around the cell that contains the particle
        for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
            for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
                for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                    let cell_number = start_cell_number + vec3i(offsetX, offsetY, offsetZ);
                    if !cellNumberInGridRange(cell_number) { continue; }

                    let cell_index = calculateCellIndexFromCellNumber(cell_number);
                    if cell_index == GRID_HASH_MAP_BLOCK_INDEX_EMPTY { continue; }
                    
                    // w
                    let cell_weight = cell_weights[u32(offsetX + 1)].x
                        * cell_weights[u32(offsetY + 1)].y
                        * cell_weights[u32(offsetZ + 1)].z;

                    // ∇w (gradient wrt fractional pos)
                    let cell_weight_gradient = vec3f(
                        cell_weights_deriv[u32(offsetX + 1)].x * cell_weights[u32(offsetY + 1)].y * cell_weights[u32(offsetZ + 1)].z,
                        cell_weights[u32(offsetX + 1)].x * cell_weights_deriv[u32(offsetY + 1)].y * cell_weights[u32(offsetZ + 1)].z,
                        cell_weights[u32(offsetX + 1)].x * cell_weights[u32(offsetY + 1)].y * cell_weights_deriv[u32(offsetZ + 1)].z
                    ) / cell_dims;
                    
                    // f = -V  Pᵀ  ∇w
                    let stress_force = -particleVolume * stressTranspose * cell_weight_gradient;

                    // p = m v
                    let particle_current_momentum = particle.mass * particle.vel;
                    // dp = F dt
                    let stress_momentum = stress_force * uniforms.simulationTimestep;
                    
                    let momentum = cell_weight * (particle_current_momentum) + stress_momentum;

                    atomicAdd(&grid_momentum_x[cell_index], i32(momentum.x * uniforms.fixedPointScale));
                    atomicAdd(&grid_momentum_y[cell_index], i32(momentum.y * uniforms.fixedPointScale));
                    atomicAdd(&grid_momentum_z[cell_index], i32(momentum.z * uniforms.fixedPointScale));
                    atomicAdd(&grid_mass[cell_index], i32(cell_weight * particle.mass * uniforms.fixedPointScale));
                }
            }
        }
    }

    else {
        let particle_cell_pos = vec3f(start_cell_number) + cell_frac_pos - 0.5;

        for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
            for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
                for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                    let cell_number = start_cell_number + vec3i(offsetX, offsetY, offsetZ);
                    if !cellNumberInGridRange(cell_number) { continue; }

                    let cell_index = calculateCellIndexFromCellNumber(cell_number);
                    if cell_index == GRID_HASH_MAP_BLOCK_INDEX_EMPTY { continue; }
                    
                    // w
                    let cell_weight = cell_weights[u32(offsetX + 1)].x
                        * cell_weights[u32(offsetY + 1)].y
                        * cell_weights[u32(offsetZ + 1)].z;


                    let weighted_mass = cell_weight * particle.mass;

                    let cell_particle_offset = vec3f(cell_number) - particle_cell_pos;
                    let affine_displacement = particle.deformation_displacement * cell_particle_offset;

                    let momentum = weighted_mass * (particle.pos_displacement + affine_displacement) / uniforms.simulationTimestep;


                    atomicAdd(&grid_momentum_x[cell_index], i32(momentum.x * uniforms.fixedPointScale));
                    atomicAdd(&grid_momentum_y[cell_index], i32(momentum.y * uniforms.fixedPointScale));
                    atomicAdd(&grid_momentum_z[cell_index], i32(momentum.z * uniforms.fixedPointScale));
                    atomicAdd(&grid_mass[cell_index], i32(weighted_mass * uniforms.fixedPointScale));
                }
            }
        }
    }
}