@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid: SparseGridStorage;
// gridUpdate converts these buffers from fixed-point momentum into fixed-point
// velocity for G2P. They become momentum again only after the next clear/P2G.
@group(1) @binding(4) var<storage, read_write> grid_momentum_x: array<i32>;
@group(1) @binding(5) var<storage, read_write> grid_momentum_y: array<i32>;
@group(1) @binding(6) var<storage, read_write> grid_momentum_z: array<i32>;

@group(2) @binding(0) var<storage, read_write> particle_data: array<ParticleData>;

override N_PARTICLES: u32 = 0u;
override USE_MLS_MPM: u32 = 0u;

@compute
@workgroup_size(256)
fn doGridToParticle(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let thread_index = gid.x;
    if thread_index >= N_PARTICLES { return; }

    let particle_pos = particle_data[thread_index].pos;
    let particle_grid_coord = calculateGridCoordinate(particle_pos);
    if !gridCoordinateCanTouchGrid(particle_grid_coord) { return; }

    let start_cell_number = vec3i(floor(particle_grid_coord));

    var block_neighborhood = loadThreeCellStencilBlockNeighborhood(start_cell_number);
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
        var new_particle_velocity_fixed_units = vec3f(0);
        var total_velocity_gradient_fixed_units = mat3x3f();
        var has_velocity_contribution = false;
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
                    if cell_weight == 0.0 { continue; }

                    let cell_number = vec3i(cell_x, cell_y, cell_z);
                    let cell_index_within_block = stencil_cell_x_bits[x_index] | cell_yz_bits;

                    let cell_index = calculateCellIndexFromLoadedBlockNeighborhoodWithLocalIndex(
                        cell_number,
                        cell_index_within_block,
                        &block_neighborhood,
                    );
                    if cell_index == GRID_BLOCK_INDEX_EMPTY { continue; }
                    
                    let cell_velocity_fixed = vec3i(
                        grid_momentum_x[cell_index],
                        grid_momentum_y[cell_index],
                        grid_momentum_z[cell_index],
                    );
                    if all(cell_velocity_fixed == vec3i(0)) { continue; }
                    has_velocity_contribution = true;

                    let cell_velocity_fixed_f = vec3f(cell_velocity_fixed);

                    new_particle_velocity_fixed_units += cell_weight * cell_velocity_fixed_f;

                    let weighted_cell_velocity_x_columns = wx * cell_velocity_fixed_f;

                    total_velocity_gradient_fixed_units += mat3x3f(
                        (stencil_weight_deriv_x[x_index] * wy_wz) * cell_velocity_fixed_f,
                        dwy_wz * weighted_cell_velocity_x_columns,
                        wy_dwz * weighted_cell_velocity_x_columns,
                    );
                }
            }
        }

        let new_particle_velocity = clampVec3LengthNoSanitizeWithMaxSquared(
            new_particle_velocity_fixed_units * INV_FIXED_POINT_SCALE,
            maxStableParticleSpeed(),
            maxStableParticleSpeedSquared(),
        );
        
        // Defer position and deformation update to integrateParticles
        particle_data[thread_index].vel = new_particle_velocity;
        var deformation_displacement = mat3x3f();
        if has_velocity_contribution {
            deformation_displacement = sanitizeNonZeroDeformationDelta(
                total_velocity_gradient_fixed_units * uniforms.explicitDeformationGradientScale
            );
        }
        particle_data[thread_index].deformation_displacement = deformation_displacement;
    }

    else {
        // enumerate the 3x3 neighborhood of cells around the cell that contains the particle
        var new_particle_velocity_fixed_units = vec3f(0);
        var B_fixed_units = mat3x3f(
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        );
        var has_velocity_contribution = false;
        let stencil_offset_x = vec3f(-0.5, 0.5, 1.5) - vec3f(cell_frac_pos.x);
        let stencil_offset_y = vec3f(-0.5, 0.5, 1.5) - vec3f(cell_frac_pos.y);
        let stencil_offset_z = vec3f(-0.5, 0.5, 1.5) - vec3f(cell_frac_pos.z);

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
                    if cell_weight == 0.0 { continue; }

                    let cell_number = vec3i(cell_x, cell_y, cell_z);
                    let cell_index_within_block = stencil_cell_x_bits[x_index] | cell_yz_bits;

                    let cell_index = calculateCellIndexFromLoadedBlockNeighborhoodWithLocalIndex(
                        cell_number,
                        cell_index_within_block,
                        &block_neighborhood,
                    );
                    if cell_index == GRID_BLOCK_INDEX_EMPTY { continue; }
                    
                    let cell_velocity_fixed = vec3i(
                        grid_momentum_x[cell_index],
                        grid_momentum_y[cell_index],
                        grid_momentum_z[cell_index],
                    );
                    if all(cell_velocity_fixed == vec3i(0)) { continue; }
                    has_velocity_contribution = true;

                    let cell_velocity_fixed_f = vec3f(cell_velocity_fixed);

                    let weighted_cell_velocity_fixed_units = cell_weight * cell_velocity_fixed_f;
                    new_particle_velocity_fixed_units += weighted_cell_velocity_fixed_units;

                    B_fixed_units += mat3x3f(
                        weighted_cell_velocity_fixed_units * stencil_offset_x[x_index],
                        weighted_cell_velocity_fixed_units * stencil_offset_y_value,
                        weighted_cell_velocity_fixed_units * stencil_offset_z_value,
                    );
                }
            }
        }

        let new_particle_velocity = clampVec3LengthNoSanitizeWithMaxSquared(
            new_particle_velocity_fixed_units * INV_FIXED_POINT_SCALE,
            maxStableParticleSpeed(),
            maxStableParticleSpeedSquared(),
        );
        var deformation_displacement = mat3x3f();
        if has_velocity_contribution {
            deformation_displacement = sanitizeNonZeroDeformationDelta(scaleMatrixColumns(
                B_fixed_units,
                uniforms.mlsDeformationGradientScale,
            ));
        }

        particle_data[thread_index].vel = new_particle_velocity;
        particle_data[thread_index].deformation_displacement = deformation_displacement;
    }
}
