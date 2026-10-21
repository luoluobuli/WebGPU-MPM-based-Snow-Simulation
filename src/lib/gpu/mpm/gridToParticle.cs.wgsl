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
    let cell_frac_pos = particle_grid_coord - vec3f(start_cell_number);
    let cell_weights = calculateQuadraticBSplineCellWeights(cell_frac_pos);
    let cell_weight_derivs = calculateQuadraticBSplineCellWeightDerivatives(cell_frac_pos);
    var block_neighborhood = loadParticleBlockNeighborhood(start_cell_number);

    var new_particle_velocity = vec3f(0.0);
    var velocity_gradient = mat3x3f();
    var b_matrix = mat3x3f();
    var has_velocity_contribution = false;

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

                let cell_velocity = grid_velocity[cell_index].velocity;
                if all(cell_velocity == vec3f(0.0)) { continue; }
                has_velocity_contribution = true;

                let cell_weight = cell_weights[x].x * (cell_weights[y].y * cell_weights[z].z);
                let weighted_cell_velocity = cell_weight * cell_velocity;
                new_particle_velocity += weighted_cell_velocity;

                if USE_MLS_MPM != 0u {
                    let cell_particle_offset = vec3f(cell_offset) + vec3f(0.5) - cell_frac_pos;
                    b_matrix += mat3x3f(
                        weighted_cell_velocity * cell_particle_offset.x,
                        weighted_cell_velocity * cell_particle_offset.y,
                        weighted_cell_velocity * cell_particle_offset.z,
                    );
                } else {
                    let cell_weight_gradient = vec3f(
                        cell_weight_derivs[x].x * cell_weights[y].y * cell_weights[z].z,
                        cell_weights[x].x * cell_weight_derivs[y].y * cell_weights[z].z,
                        cell_weights[x].x * cell_weights[y].y * cell_weight_derivs[z].z,
                    ) * uniforms.invGridCellDims;
                    velocity_gradient += mat3x3f(
                        cell_weight_gradient.x * cell_velocity,
                        cell_weight_gradient.y * cell_velocity,
                        cell_weight_gradient.z * cell_velocity,
                    );
                }
            }
        }
    }

    particle_data[thread_index].vel = clampVec3LengthWithMaxSquared(
        new_particle_velocity,
        maxFixedPointGridSpeed(),
        maxFixedPointGridSpeedSquared(),
    );

    // Defer position and deformation updates to integrateParticles.
    var deformation_displacement = mat3x3f();
    if has_velocity_contribution {
        if USE_MLS_MPM != 0u {
            deformation_displacement = sanitizeVelocityGradient(scaleMatrixColumns(
                b_matrix,
                uniforms.mlsDeformationGradientScale,
            ));
        } else {
            deformation_displacement = sanitizeVelocityGradient(
                velocity_gradient * uniforms.explicitDeformationGradientScale,
            );
        }
    }

    particle_flags[thread_index] = persistent_flags;
    if !mat3x3IsZero(deformation_displacement) {
        particle_data[thread_index].deformation_displacement = deformation_displacement;
        particle_flags[thread_index] |= PARTICLE_FLAG_DEFORMATION_DELTA_VALID;
    }
}
