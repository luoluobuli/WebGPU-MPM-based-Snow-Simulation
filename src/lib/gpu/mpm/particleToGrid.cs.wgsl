@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid: SparseGridStorage;

struct GridAccumulatorAtomic {
    mass: atomic<i32>,
    momentum_x: atomic<i32>,
    momentum_y: atomic<i32>,
    momentum_z: atomic<i32>,
}

@group(1) @binding(3) var<storage, read_write> grid_accumulator: array<GridAccumulatorAtomic>;

struct GridTransferAccumulatorAtomic {
    momentum_x: atomic<i32>,
    momentum_y: atomic<i32>,
    momentum_z: atomic<i32>,
    _padding: atomic<i32>,
}

@group(1) @binding(7) var<storage, read_write> grid_transfer_accumulator: array<GridTransferAccumulatorAtomic>;

@group(2) @binding(0) var<storage, read> particleDataIn: array<ParticleData>;
@group(2) @binding(2) var<storage, read> particle_flags: array<u32>;
@group(2) @binding(3) var<storage, read_write> active_block_dispatch_args: array<u32>;

override N_PARTICLES: u32 = 0u;
override USE_MLS_MPM: u32 = 0u;
override SANITIZE_PARTICLES_IN_P2G: u32 = 0u;
override WRITE_FLIP_TRANSFER: u32 = 0u;
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

fn accumulateParticleTransferMomentum(
    cell_index: u32,
    momentum_fixed_units: vec3f,
) {
    if WRITE_FLIP_TRANSFER == 0u { return; }

    let momentum_i = vec3i(momentum_fixed_units);
    atomicAdd(&grid_transfer_accumulator[cell_index].momentum_x, momentum_i.x);
    atomicAdd(&grid_transfer_accumulator[cell_index].momentum_y, momentum_i.y);
    atomicAdd(&grid_transfer_accumulator[cell_index].momentum_z, momentum_i.z);
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
    let cell_frac_pos = particle_grid_coord - vec3f(start_cell_number);
    let cell_weights = calculateQuadraticBSplineCellWeights(cell_frac_pos);
    let cell_weight_derivs = calculateQuadraticBSplineCellWeightDerivatives(cell_frac_pos);
    var block_neighborhood = loadParticleBlockNeighborhood(start_cell_number);

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

    var stress_force_matrix = mat3x3f();
    if !elastic_is_identity {
        if SANITIZE_PARTICLES_IN_P2G == 0u {
            deformation_plastic = particleDataIn[thread_index].deformationPlastic;
        }
        var shear_resistance = SHEAR_RESISTANCE; // μ
        var volumetric_resistance = VOLUME_RESISTANCE; // λ
        applyMaterialLameParameters(material, &shear_resistance, &volumetric_resistance);
        hardenLameParameters(
            material,
            deformation_plastic,
            &shear_resistance,
            &volumetric_resistance,
        );
        let stress = calculateStressFirstPiolaKirchhoffNonIdentity( // P
            deformation_elastic,
            shear_resistance,
            volumetric_resistance,
        );
        stress_force_matrix = stress * transpose(deformation_elastic);
    }

    let particle_mass_fixed_units = particle_mass * FIXED_POINT_SCALE;
    let particle_momentum_fixed_units = particle_mass_fixed_units * particle_vel; // p = m v
    let max_particle_momentum_fixed_units = particle_mass_fixed_units * maxFixedPointGridSpeed();
    let simple_transfer = elastic_is_identity && (USE_MLS_MPM == 0u || !has_deformation_delta);
    let transfer_affine_fixed_units = particle_mass_fixed_units * deformation_displacement;
    var mls_affine_fixed_units = transfer_affine_fixed_units;
    if USE_MLS_MPM != 0u && !elastic_is_identity {
        mls_affine_fixed_units += scaleMatrixColumns(
            particle_mass * stress_force_matrix,
            uniforms.mlsStressAffineScale,
        );
    }
    let stress_impulse_matrix = particle_mass
        * uniforms.explicitStressImpulseScale
        * stress_force_matrix;

    // Enumerate the 3x3x3 neighborhood around the particle's cell.
    for (var z = 0u; z < 3u; z++) {
        for (var y = 0u; y < 3u; y++) {
            for (var x = 0u; x < 3u; x++) {
                let cell_offset = vec3i(vec3u(x, y, z)) - vec3i(1);
                let cell_number = start_cell_number + cell_offset;
                let cell_index = calculateCellIndexFromLoadedBlockNeighborhood(
                    cell_number,
                    &block_neighborhood,
                );
                if cell_index == GRID_BLOCK_INDEX_EMPTY { continue; }

                let cell_weight = cell_weights[x].x * (cell_weights[y].y * cell_weights[z].z);
                var momentum_fixed_units = cell_weight * particle_momentum_fixed_units;
                var transfer_momentum_fixed_units = momentum_fixed_units;

                if !simple_transfer {
                    if USE_MLS_MPM != 0u {
                        let cell_particle_offset = (
                            vec3f(cell_offset) + vec3f(0.5) - cell_frac_pos
                        ) * uniforms.gridCellDims;
                        momentum_fixed_units = cell_weight * (
                            particle_momentum_fixed_units
                            + mls_affine_fixed_units * cell_particle_offset
                        );
                        transfer_momentum_fixed_units = cell_weight * (
                            particle_momentum_fixed_units
                            + transfer_affine_fixed_units * cell_particle_offset
                        );
                    } else {
                        // ∇w (gradient wrt world position)
                        let cell_weight_gradient = vec3f(
                            cell_weight_derivs[x].x * cell_weights[y].y * cell_weights[z].z,
                            cell_weights[x].x * cell_weight_derivs[y].y * cell_weights[z].z,
                            cell_weights[x].x * cell_weights[y].y * cell_weight_derivs[z].z,
                        ) * uniforms.invGridCellDims;
                        momentum_fixed_units += stress_impulse_matrix * cell_weight_gradient;
                    }

                    momentum_fixed_units = clampMomentumFixedUnitsForAtomic(
                        momentum_fixed_units,
                        max_particle_momentum_fixed_units,
                    );
                    transfer_momentum_fixed_units = clampMomentumFixedUnitsForAtomic(
                        transfer_momentum_fixed_units,
                        max_particle_momentum_fixed_units,
                    );
                }

                accumulateParticleToGridFixedUnits(
                    cell_index,
                    momentum_fixed_units,
                    cell_weight * particle_mass_fixed_units,
                );
                if USE_MLS_MPM != 0u || simple_transfer {
                    accumulateParticleTransferMomentum(cell_index, transfer_momentum_fixed_units);
                }
            }
        }
    }
}
