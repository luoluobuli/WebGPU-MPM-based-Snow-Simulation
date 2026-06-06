@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid: SparseGridStorage;

struct GridAccumulatorAtomic {
    mass: atomic<i32>,
    momentum_x: atomic<i32>,
    momentum_y: atomic<i32>,
    momentum_z: atomic<i32>,
}

@group(1) @binding(3) var<storage, read_write> grid_accumulator: array<GridAccumulatorAtomic>;

@group(2) @binding(0) var<storage, read> particleDataIn: array<ParticleData>;
@group(2) @binding(2) var<storage, read> particle_flags: array<u32>;
@group(2) @binding(3) var<storage, read_write> active_block_dispatch_args: array<u32>;

override N_PARTICLES: u32 = 0u;
override USE_MLS_MPM: u32 = 0u;
override SANITIZE_PARTICLES_IN_P2G: u32 = 0u;
override PARTICLE_WORKGROUP_SIZE: u32 = 256u;

fn clampMomentumFixedUnitsForAtomic(value: vec3f, max_component: f32) -> vec3f {
    return clamp(value, vec3f(-max_component), vec3f(max_component));
}

fn accumulateParticleToGridFixedUnits(
    cell_index: u32,
    momentum_fixed_units: vec3f,
    mass_fixed_units: f32,
) {
    let momentum_i = vec3i(momentum_fixed_units);
    atomicAdd(&grid_accumulator[cell_index].mass, i32(mass_fixed_units));
    atomicAdd(&grid_accumulator[cell_index].momentum_x, momentum_i.x);
    atomicAdd(&grid_accumulator[cell_index].momentum_y, momentum_i.y);
    atomicAdd(&grid_accumulator[cell_index].momentum_z, momentum_i.z);
}

fn writeActiveBlockDispatchArgs() {
    let count = min(atomicLoad(&sparse_grid.n_allocated_blocks), N_MAX_ACTIVE_BLOCKS);

    if count == 0u {
        active_block_dispatch_args[0] = 1u;
        active_block_dispatch_args[1] = 1u;
        active_block_dispatch_args[2] = 1u;
        active_block_dispatch_args[3] = 0u;
        return;
    }

    active_block_dispatch_args[0] = min(count, 256u);
    active_block_dispatch_args[1] = (count + 255u) / 256u;
    active_block_dispatch_args[2] = 1u;
    active_block_dispatch_args[3] = count;
}

@compute
@workgroup_size(PARTICLE_WORKGROUP_SIZE)
fn doParticleToGrid(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let thread_index = gid.x;
    if thread_index == 0u {
        writeActiveBlockDispatchArgs();
    }
    if thread_index >= N_PARTICLES { return; }

    let flags = particle_flags[thread_index];
    let material = particleMaterial(flags);
    var particle_pos = particleDataIn[thread_index].pos;
    var particle_mass = DEFAULT_PARTICLE_MASS;
    var particle_vel = vec3f(0.0);
    var deformation_elastic = IDENTITY_MAT3;
    var deformation_plastic = mat3x3f();
    var deformation_displacement = mat3x3f();
    var elastic_is_identity = (flags & PARTICLE_FLAG_ELASTIC_NON_IDENTITY) == 0u;

    // Particle data is sanitized at initialization and by integrateParticles at
    // both ends of each substep. P2G is on the hot path and only reads it.
    if SANITIZE_PARTICLES_IN_P2G != 0u {
        var particle = particleDataIn[thread_index];
        sanitizeParticle(&particle);
        particle_pos = particle.pos;
        particle_mass = particle.mass;
        particle_vel = particle.vel;
        deformation_elastic = particle.deformationElastic;
        deformation_plastic = particle.deformationPlastic;
        deformation_displacement = particle.deformation_displacement;
        elastic_is_identity = mat3x3IsIdentity(deformation_elastic);
    }

    let particle_grid_coord = calculateGridCoordinate(particle_pos);
    if !gridCoordinateCanTouchGrid(particle_grid_coord) { return; }

    if SANITIZE_PARTICLES_IN_P2G == 0u {
        particle_vel = particleDataIn[thread_index].vel;
        if !elastic_is_identity {
            deformation_elastic = particleDataIn[thread_index].deformationElastic;
        }
    }

    let start_cell_number = vec3i(floor(particle_grid_coord));

    var block_neighborhood = loadThreeCellStencilBlockNeighborhoodForGeneration(
        start_cell_number,
        sparse_grid.current_generation,
    );
    let cell_frac_pos = particle_grid_coord - vec3f(start_cell_number);
    let cell_weights = calculateQuadraticBSplineCellWeightVectors(cell_frac_pos);
    let stencil_cell_x = start_cell_number.x + vec3i(-1i, 0i, 1i);
    let stencil_cell_x_bits = vec3u(stencil_cell_x & vec3i(i32(BLOCK_MASK)));
    let stencil_weight_x = cell_weights.x;
    let stencil_weight_y = cell_weights.y;
    let stencil_weight_z = cell_weights.z;

    var has_deformation_delta = false;
    if USE_MLS_MPM != 0u {
        if SANITIZE_PARTICLES_IN_P2G == 0u {
            has_deformation_delta = (flags & PARTICLE_FLAG_DEFORMATION_DELTA_VALID) != 0u;
            if has_deformation_delta {
                deformation_displacement = sanitizeVelocityGradient(
                    particleDataIn[thread_index].deformation_displacement,
                );
            }
        } else {
            has_deformation_delta = !mat3x3IsZero(deformation_displacement);
        }
    }

    let particle_mass_fixed_units = particle_mass * FIXED_POINT_SCALE;
    let particle_momentum_fixed_units = particle_mass_fixed_units * particle_vel;

    if elastic_is_identity && (USE_MLS_MPM == 0u || !has_deformation_delta) {
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

                    accumulateParticleToGridFixedUnits(
                        cell_index,
                        cell_weight * particle_momentum_fixed_units,
                        cell_weight * particle_mass_fixed_units,
                    );
                }
            }
        }
        return;
    }

    var stress_force_matrix = mat3x3f();
    if !elastic_is_identity {
        if SANITIZE_PARTICLES_IN_P2G == 0u {
            deformation_plastic = particleDataIn[thread_index].deformationPlastic;
        }
        var shear_resistance = SHEAR_RESISTANCE;
        var volumetric_resistance = VOLUME_RESISTANCE;
        applyMaterialLameParameters(material, &shear_resistance, &volumetric_resistance);
        hardenLameParameters(material, deformation_plastic, &shear_resistance, &volumetric_resistance);
        let stress = calculateStressFirstPiolaKirchhoffNonIdentity(
            deformation_elastic,
            shear_resistance,
            volumetric_resistance,
        );
        stress_force_matrix = stress * transpose(deformation_elastic);
    }

    let max_particle_momentum_fixed_units = particle_mass_fixed_units * maxFixedPointGridSpeed();

    if USE_MLS_MPM != 0u {
        let affine_velocity = deformation_displacement;
        var mls_affine_fixed_units = particle_mass_fixed_units * affine_velocity;
        let grid_cell_dims = uniforms.gridCellDims;
        let stencil_offset_x = (vec3f(-0.5, 0.5, 1.5) - vec3f(cell_frac_pos.x)) * grid_cell_dims.x;
        let stencil_offset_y = (vec3f(-0.5, 0.5, 1.5) - vec3f(cell_frac_pos.y)) * grid_cell_dims.y;
        let stencil_offset_z = (vec3f(-0.5, 0.5, 1.5) - vec3f(cell_frac_pos.z)) * grid_cell_dims.z;
        if !elastic_is_identity {
            let mls_stress_affine_fixed_units = scaleMatrixColumns(
                particle_mass * stress_force_matrix,
                uniforms.mlsStressAffineScale,
            );
            mls_affine_fixed_units = mls_affine_fixed_units + mls_stress_affine_fixed_units;
        }

        for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
            let cell_z = start_cell_number.z + offsetZ;
            let z_index = u32(offsetZ + 1i);
            let cell_z_bits = u32(cell_z & i32(BLOCK_MASK)) << (LOG_BLOCK_SIZE * 2u);
            let wz = stencil_weight_z[z_index];
            let mls_affine_z_contribution = mls_affine_fixed_units[2] * stencil_offset_z[z_index];
            for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
                let cell_y = start_cell_number.y + offsetY;
                let y_index = u32(offsetY + 1i);
                let cell_yz_bits = (u32(cell_y & i32(BLOCK_MASK)) << LOG_BLOCK_SIZE) | cell_z_bits;
                let wy_wz = stencil_weight_y[y_index] * wz;
                let mls_affine_y_contribution = mls_affine_fixed_units[1] * stencil_offset_y[y_index];
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

                    let weighted_mass_fixed_units = cell_weight * particle_mass_fixed_units;

                    let affine_offset_fixed_units = mls_affine_fixed_units[0] * stencil_offset_x[x_index]
                        + mls_affine_y_contribution
                        + mls_affine_z_contribution;
                    let momentum_fixed_units = clampMomentumFixedUnitsForAtomic(
                        cell_weight * (particle_momentum_fixed_units + affine_offset_fixed_units),
                        max_particle_momentum_fixed_units
                    );

                    accumulateParticleToGridFixedUnits(cell_index, momentum_fixed_units, weighted_mass_fixed_units);
                }
            }
        }
    }
    else {
        let cell_weight_derivs = calculateQuadraticBSplineCellWeightDerivativeVectors(
            cell_frac_pos,
            uniforms.invGridCellDims,
        );
        let stencil_weight_deriv_x = cell_weight_derivs.x;
        let stencil_weight_deriv_y = cell_weight_derivs.y;
        let stencil_weight_deriv_z = cell_weight_derivs.z;
        let stress_impulse_matrix_fixed_units = particle_mass * uniforms.explicitStressImpulseScale * stress_force_matrix;
        let stress_impulse_matrix_x_fixed_units = stress_impulse_matrix_fixed_units[0];
        let stress_impulse_matrix_y_fixed_units = stress_impulse_matrix_fixed_units[1];
        let stress_impulse_matrix_z_fixed_units = stress_impulse_matrix_fixed_units[2];

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
                let stress_impulse_yz_fixed_units = stress_impulse_matrix_y_fixed_units * dwy_wz
                    + stress_impulse_matrix_z_fixed_units * wy_dwz;
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

                    let weighted_mass_fixed_units = cell_weight * particle_mass_fixed_units;

                    let stress_impulse_fixed_units =
                        stress_impulse_matrix_x_fixed_units * (stencil_weight_deriv_x[x_index] * wy_wz)
                        + stress_impulse_yz_fixed_units * wx;
                    let momentum_fixed_units = clampMomentumFixedUnitsForAtomic(
                        cell_weight * particle_momentum_fixed_units + stress_impulse_fixed_units,
                        max_particle_momentum_fixed_units
                    );

                    accumulateParticleToGridFixedUnits(cell_index, momentum_fixed_units, weighted_mass_fixed_units);
                }
            }
        }
    }
}
