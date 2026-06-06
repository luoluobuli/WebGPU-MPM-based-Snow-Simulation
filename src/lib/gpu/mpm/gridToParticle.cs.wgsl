@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid: SparseGridStorage;

struct GridVelocity {
    velocity: vec3f,
}

@group(1) @binding(7) var<storage, read_write> grid_velocity: array<GridVelocity>;

@group(2) @binding(0) var<storage, read_write> particle_data: array<ParticleData>;
@group(2) @binding(2) var<storage, read_write> particle_flags: array<u32>;

override N_PARTICLES: u32 = 0u;
override USE_MLS_MPM: u32 = 0u;
override PARTICLE_WORKGROUP_SIZE: u32 = 256u;

@compute
@workgroup_size(PARTICLE_WORKGROUP_SIZE)
fn doGridToParticle(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let thread_index = gid.x;
    if thread_index >= N_PARTICLES { return; }

    let flags = particle_flags[thread_index];
    let persistent_flags = particlePersistentFlags(flags);
    let particle_pos = particle_data[thread_index].pos;
    let particle_grid_coord = calculateGridCoordinate(particle_pos);
    if !gridCoordinateCanTouchGrid(particle_grid_coord) {
        particle_flags[thread_index] = persistent_flags;
        return;
    }

    let start_cell_number = vec3i(floor(particle_grid_coord));

    let current_generation = sparse_grid.current_generation;
    var block_neighborhood = loadThreeCellStencilBlockNeighborhoodForGeneration(
        start_cell_number,
        current_generation,
    );
    let cell_frac_pos = particle_grid_coord - vec3f(start_cell_number);
    let cell_weights = calculateQuadraticBSplineCellWeightVectors(cell_frac_pos);
    let stencil_cell_x = start_cell_number.x + vec3i(-1i, 0i, 1i);
    let stencil_cell_x_bits = vec3u(stencil_cell_x & vec3i(i32(BLOCK_MASK)));
    let stencil_weight_x = cell_weights.x;
    let stencil_weight_y = cell_weights.y;
    let stencil_weight_z = cell_weights.z;

    if USE_MLS_MPM == 0u {
        let cell_weight_derivs = calculateQuadraticBSplineCellWeightDerivativeVectors(
            cell_frac_pos,
            uniforms.invGridCellDims,
        );
        let stencil_weight_deriv_x = cell_weight_derivs.x;
        let stencil_weight_deriv_y = cell_weight_derivs.y;
        let stencil_weight_deriv_z = cell_weight_derivs.z;

        // enumerate the 3x3 neighborhood of cells around the cell that contains the particle
        var new_particle_velocity_in = vec3f(0);
        var total_velocity_gradient = mat3x3f();
        var has_velocity_contribution = false;
        let single_block_cell_index_base = block_neighborhood.single_block_cell_index_base;
        if single_block_cell_index_base != GRID_BLOCK_INDEX_EMPTY {
            for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
                let cell_z = start_cell_number.z + offsetZ;
                let z_index = u32(offsetZ + 1i);
                let cell_z_bits = u32(cell_z & i32(BLOCK_MASK)) << (LOG_BLOCK_SIZE * 2u);
                let wz = stencil_weight_z[z_index];
                let dwz = stencil_weight_deriv_z[z_index];
                for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
                    let cell_y = start_cell_number.y + offsetY;
                    let y_index = u32(offsetY + 1i);
                    let cell_yz_bits = (u32(cell_y & i32(BLOCK_MASK)) << LOG_BLOCK_SIZE) | cell_z_bits;
                    let wy = stencil_weight_y[y_index];
                    let dwy = stencil_weight_deriv_y[y_index];
                    let wy_wz = wy * wz;
                    let dwy_wz = dwy * wz;
                    let wy_dwz = wy * dwz;
                    for (var x_index = 0u; x_index < 3u; x_index++) {
                        let wx = stencil_weight_x[x_index];
                        let cell_weight = wx * wy_wz;
                        let cell_index_within_block = stencil_cell_x_bits[x_index] | cell_yz_bits;
                        let cell_index = single_block_cell_index_base + cell_index_within_block;

                        let cell_velocity = grid_velocity[cell_index].velocity;
                        if all(cell_velocity == vec3f(0.0)) { continue; }
                        has_velocity_contribution = true;

                        new_particle_velocity_in += cell_weight * cell_velocity;

                        let weighted_cell_velocity_x_columns = wx * cell_velocity;

                        total_velocity_gradient += mat3x3f(
                            (stencil_weight_deriv_x[x_index] * wy_wz) * cell_velocity,
                            dwy_wz * weighted_cell_velocity_x_columns,
                            wy_dwz * weighted_cell_velocity_x_columns,
                        );
                    }
                }
            }
        } else {
            for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
                let cell_z = start_cell_number.z + offsetZ;
                let z_index = u32(offsetZ + 1i);
                let cell_z_bits = u32(cell_z & i32(BLOCK_MASK)) << (LOG_BLOCK_SIZE * 2u);
                let wz = stencil_weight_z[z_index];
                let dwz = stencil_weight_deriv_z[z_index];
                for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
                    let cell_y = start_cell_number.y + offsetY;
                    let y_index = u32(offsetY + 1i);
                    let cell_yz_bits = (u32(cell_y & i32(BLOCK_MASK)) << LOG_BLOCK_SIZE) | cell_z_bits;
                    let wy = stencil_weight_y[y_index];
                    let dwy = stencil_weight_deriv_y[y_index];
                    let wy_wz = wy * wz;
                    let dwy_wz = dwy * wz;
                    let wy_dwz = wy * dwz;
                    for (var x_index = 0u; x_index < 3u; x_index++) {
                        let cell_x = stencil_cell_x[x_index];
                        let wx = stencil_weight_x[x_index];
                        let cell_weight = wx * wy_wz;

                        let cell_index_within_block = stencil_cell_x_bits[x_index] | cell_yz_bits;

                        let cell_index = calculateCellIndexFromLoadedBlockNeighborhoodFast(
                            cell_x,
                            cell_y,
                            cell_z,
                            cell_index_within_block,
                            &block_neighborhood,
                        );
                        if cell_index == GRID_BLOCK_INDEX_EMPTY { continue; }

                        let cell_velocity = grid_velocity[cell_index].velocity;
                        if all(cell_velocity == vec3f(0.0)) { continue; }
                        has_velocity_contribution = true;

                        new_particle_velocity_in += cell_weight * cell_velocity;

                        let weighted_cell_velocity_x_columns = wx * cell_velocity;

                        total_velocity_gradient += mat3x3f(
                            (stencil_weight_deriv_x[x_index] * wy_wz) * cell_velocity,
                            dwy_wz * weighted_cell_velocity_x_columns,
                            wy_dwz * weighted_cell_velocity_x_columns,
                        );
                    }
                }
            }
        }

        let new_particle_velocity = clampVec3LengthWithMaxSquared(
            new_particle_velocity_in,
            maxFixedPointGridSpeed(),
            maxFixedPointGridSpeedSquared(),
        );
        
        // Defer position and deformation update to integrateParticles
        particle_data[thread_index].vel = new_particle_velocity;
        var deformation_displacement = mat3x3f();
        if has_velocity_contribution {
            deformation_displacement = sanitizeVelocityGradient(
                total_velocity_gradient * uniforms.explicitDeformationGradientScale
            );
        }
        if mat3x3IsZero(deformation_displacement) {
            particle_flags[thread_index] = persistent_flags;
        } else {
            particle_data[thread_index].deformation_displacement = deformation_displacement;
            particle_flags[thread_index] = persistent_flags | PARTICLE_FLAG_DEFORMATION_DELTA_VALID;
        }
    }

    else {
        // enumerate the 3x3 neighborhood of cells around the cell that contains the particle
        var new_particle_velocity_in = vec3f(0);
        var B = mat3x3f(
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        );
        var has_velocity_contribution = false;
        let stencil_offset_x = vec3f(-0.5, 0.5, 1.5) - vec3f(cell_frac_pos.x);
        let stencil_offset_y = vec3f(-0.5, 0.5, 1.5) - vec3f(cell_frac_pos.y);
        let stencil_offset_z = vec3f(-0.5, 0.5, 1.5) - vec3f(cell_frac_pos.z);

        let single_block_cell_index_base = block_neighborhood.single_block_cell_index_base;
        if single_block_cell_index_base != GRID_BLOCK_INDEX_EMPTY {
            for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
                let cell_z = start_cell_number.z + offsetZ;
                let z_index = u32(offsetZ + 1i);
                let cell_z_bits = u32(cell_z & i32(BLOCK_MASK)) << (LOG_BLOCK_SIZE * 2u);
                let wz = stencil_weight_z[z_index];
                for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
                    let cell_y = start_cell_number.y + offsetY;
                    let y_index = u32(offsetY + 1i);
                    let cell_yz_bits = (u32(cell_y & i32(BLOCK_MASK)) << LOG_BLOCK_SIZE) | cell_z_bits;
                    let wy_wz = stencil_weight_y[y_index] * wz;
                    let stencil_offset_y_value = stencil_offset_y[y_index];
                    let stencil_offset_z_value = stencil_offset_z[z_index];
                    for (var x_index = 0u; x_index < 3u; x_index++) {
                        let cell_weight = stencil_weight_x[x_index] * wy_wz;
                        let cell_index_within_block = stencil_cell_x_bits[x_index] | cell_yz_bits;
                        let cell_index = single_block_cell_index_base + cell_index_within_block;

                        let cell_velocity = grid_velocity[cell_index].velocity;
                        if all(cell_velocity == vec3f(0.0)) { continue; }
                        has_velocity_contribution = true;

                        let weighted_cell_velocity = cell_weight * cell_velocity;
                        new_particle_velocity_in += weighted_cell_velocity;

                        B += mat3x3f(
                            weighted_cell_velocity * stencil_offset_x[x_index],
                            weighted_cell_velocity * stencil_offset_y_value,
                            weighted_cell_velocity * stencil_offset_z_value,
                        );
                    }
                }
            }
        } else {
            for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
                let cell_z = start_cell_number.z + offsetZ;
                let z_index = u32(offsetZ + 1i);
                let cell_z_bits = u32(cell_z & i32(BLOCK_MASK)) << (LOG_BLOCK_SIZE * 2u);
                let wz = stencil_weight_z[z_index];
                for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
                    let cell_y = start_cell_number.y + offsetY;
                    let y_index = u32(offsetY + 1i);
                    let cell_yz_bits = (u32(cell_y & i32(BLOCK_MASK)) << LOG_BLOCK_SIZE) | cell_z_bits;
                    let wy_wz = stencil_weight_y[y_index] * wz;
                    let stencil_offset_y_value = stencil_offset_y[y_index];
                    let stencil_offset_z_value = stencil_offset_z[z_index];
                    for (var x_index = 0u; x_index < 3u; x_index++) {
                        let cell_x = stencil_cell_x[x_index];
                        let cell_weight = stencil_weight_x[x_index] * wy_wz;

                        let cell_index_within_block = stencil_cell_x_bits[x_index] | cell_yz_bits;

                        let cell_index = calculateCellIndexFromLoadedBlockNeighborhoodFast(
                            cell_x,
                            cell_y,
                            cell_z,
                            cell_index_within_block,
                            &block_neighborhood,
                        );
                        if cell_index == GRID_BLOCK_INDEX_EMPTY { continue; }

                        let cell_velocity = grid_velocity[cell_index].velocity;
                        if all(cell_velocity == vec3f(0.0)) { continue; }
                        has_velocity_contribution = true;

                        let weighted_cell_velocity = cell_weight * cell_velocity;
                        new_particle_velocity_in += weighted_cell_velocity;

                        B += mat3x3f(
                            weighted_cell_velocity * stencil_offset_x[x_index],
                            weighted_cell_velocity * stencil_offset_y_value,
                            weighted_cell_velocity * stencil_offset_z_value,
                        );
                    }
                }
            }
        }

        let new_particle_velocity = clampVec3LengthWithMaxSquared(
            new_particle_velocity_in,
            maxFixedPointGridSpeed(),
            maxFixedPointGridSpeedSquared(),
        );
        var deformation_displacement = mat3x3f();
        if has_velocity_contribution {
            deformation_displacement = sanitizeVelocityGradient(scaleMatrixColumns(
                B,
                uniforms.mlsDeformationGradientScale,
            ));
        }

        particle_data[thread_index].vel = new_particle_velocity;
        if mat3x3IsZero(deformation_displacement) {
            particle_flags[thread_index] = persistent_flags;
        } else {
            particle_data[thread_index].deformation_displacement = deformation_displacement;
            particle_flags[thread_index] = persistent_flags | PARTICLE_FLAG_DEFORMATION_DELTA_VALID;
        }
    }
}
