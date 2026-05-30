@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid: SparseGridStorage;
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

    var particle = particleDataIn[thread_index];
    sanitizeParticle(&particle);
    if !particlePositionCanTouchGrid(particle.pos) { return; }

    let start_cell_number = calculateCellNumber(particle.pos);
    let cell_frac_pos = calculateFractionalPosFromCellMin(particle.pos, start_cell_number);
    let cell_weights = calculateQuadraticBSplineCellWeights(cell_frac_pos);
    let cell_weights_deriv = calculateQuadraticBSplineCellWeightDerivatives(cell_frac_pos);

    var shear_resistance = SHEAR_RESISTANCE;
    var volumetric_resistance = VOLUME_RESISTANCE;
    hardenLameParameters(particle.deformationPlastic, &shear_resistance, &volumetric_resistance);
    let stress = calculateStressFirstPiolaKirchhoff(
        particle.deformationElastic,
        shear_resistance,
        volumetric_resistance,
    );
    let stress_force_matrix = stress * transpose(particle.deformationElastic);

    const DENSITY_KG_PER_M3 = 400.;
    const INVERSE_DENSITY = 1. / DENSITY_KG_PER_M3;
    let particle_volume = particle.mass * INVERSE_DENSITY;

    let affine_velocity = sanitizeDeformationDelta(particle.deformation_displacement) * (1.0 / uniforms.simulationTimestep);
    let mls_stress_affine = scaleMatrixColumns(
        -4.0 * particle_volume * uniforms.simulationTimestep * stress_force_matrix,
        1.0 / (uniforms.gridCellDims * uniforms.gridCellDims),
    );
    let mls_affine = mls_stress_affine + particle.mass * affine_velocity;

    for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
        for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
            for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                let cell_number = start_cell_number + vec3i(offsetX, offsetY, offsetZ);
                if !cellNumberInGridRange(cell_number) { continue; }

                let cell_index = calculateCellIndexFromCellNumber(cell_number);
                if cell_index == GRID_HASH_MAP_BLOCK_INDEX_EMPTY { continue; }

                let cell_weight = cell_weights[u32(offsetX + 1)].x
                    * cell_weights[u32(offsetY + 1)].y
                    * cell_weights[u32(offsetZ + 1)].z;

                var momentum: vec3f;
                if uniforms.use_mls_mpm != 0u {
                    let cell_particle_offset = calculateCellWorldOffsetFromParticle(cell_number, particle.pos);
                    momentum = cell_weight * (particle.mass * particle.vel + mls_affine * cell_particle_offset);
                }
                else {
                    let cell_weight_gradient = vec3f(
                        cell_weights_deriv[u32(offsetX + 1)].x * cell_weights[u32(offsetY + 1)].y * cell_weights[u32(offsetZ + 1)].z,
                        cell_weights[u32(offsetX + 1)].x * cell_weights_deriv[u32(offsetY + 1)].y * cell_weights[u32(offsetZ + 1)].z,
                        cell_weights[u32(offsetX + 1)].x * cell_weights[u32(offsetY + 1)].y * cell_weights_deriv[u32(offsetZ + 1)].z,
                    ) / uniforms.gridCellDims;

                    let stress_force = -particle_volume * stress_force_matrix * cell_weight_gradient;
                    momentum = cell_weight * particle.mass * particle.vel + stress_force * uniforms.simulationTimestep;
                }

                momentum = clampVec3Length(momentum, particle.mass * maxStableParticleSpeed());

                atomicAdd(&grid_momentum_x[cell_index], toFixedPointI32(momentum.x));
                atomicAdd(&grid_momentum_y[cell_index], toFixedPointI32(momentum.y));
                atomicAdd(&grid_momentum_z[cell_index], toFixedPointI32(momentum.z));
                atomicAdd(&grid_mass[cell_index], toFixedPointI32(cell_weight * particle.mass));
            }
        }
    }
}
