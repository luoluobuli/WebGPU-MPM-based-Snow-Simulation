// pbmpm.wgsl - Fused SolveConstraints + P2G shader with dense grid
// Uses O(1) direct grid indexing instead of hash map lookups

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> bukkitCounts: array<u32>;
@group(1) @binding(1) var<storage, read_write> bukkitDispatch: array<u32>;
@group(1) @binding(2) var<storage, read_write> bukkitThreadData: array<BukkitThreadData>;
@group(1) @binding(3) var<storage, read_write> grid_mass: array<atomic<i32>>;
@group(1) @binding(4) var<storage, read_write> grid_momentum_x: array<atomic<i32>>;
@group(1) @binding(5) var<storage, read_write> grid_momentum_y: array<atomic<i32>>;
@group(1) @binding(6) var<storage, read_write> grid_momentum_z: array<atomic<i32>>;

@group(2) @binding(0) var<storage, read_write> particle_data: array<ParticleData>;
@group(2) @binding(1) var<storage, read_write> sortedParticleIndices: array<u32>;

fn solveParticleConstraints(particle: ptr<function, ParticleData>) {
    // this is the inverse of the formula used to integrate deformation
    let trial_deformation_elastic = (IDENTITY_MAT3 + (*particle).deformation_displacement) * (*particle).deformationElastic; 
    let trial_rotation = calculatePolarDecompositionRotation(trial_deformation_elastic);

    let target_volume = determinant(trial_deformation_elastic);
    
    let volume_scale = pow(abs(target_volume), -0.3333333);
    let target_scaled = trial_rotation * volume_scale;

    let blend_factor = 0.95;
    let target_blended = blend_factor * target_scaled + (1 - blend_factor) * trial_rotation;

    let corrected_deformation_displacement = target_blended * mat3x3Inverse((*particle).deformationElastic) - IDENTITY_MAT3;

    let volumeScaleFac = determinant((*particle).deformationPlastic); // J
    
    let deformation_displacement_diff = corrected_deformation_displacement - (*particle).deformation_displacement;

    const HARDENING_COEFFICIENT = 20.;
    let elasticity_relaxation = 0.5 + 0.5 * (1 - exp(-HARDENING_COEFFICIENT / volumeScaleFac));
    
    (*particle).deformation_displacement += elasticity_relaxation * deformation_displacement_diff;
}

fn transferParticlesToGrid(particle: ptr<function, ParticleData>) {
    let start_cell_number = calculateCellNumber((*particle).pos);
    let cell_frac_pos = calculateFractionalPosFromCellMin((*particle).pos, start_cell_number);
    let cell_weights = calculateQuadraticBSplineCellWeights(cell_frac_pos);
    
    let particle_cell_pos = vec3f(start_cell_number) + cell_frac_pos - 0.5;

    for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
        for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
            for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                let cell_number = start_cell_number + vec3i(offsetX, offsetY, offsetZ);
                if !cellNumberInGridRange(cell_number) { continue; }

                // O(1) lookup - no hash map!
                let cell_index = cellToGridIndex(cell_number);
                if cell_index == 0xFFFFFFFFu { continue; }
                
                // w
                let cell_weight = cell_weights[u32(offsetX + 1)].x
                    * cell_weights[u32(offsetY + 1)].y
                    * cell_weights[u32(offsetZ + 1)].z;

                let weighted_mass = cell_weight * (*particle).mass;

                let cell_particle_offset = vec3f(cell_number) - particle_cell_pos;
                let affine_displacement = (*particle).deformation_displacement * cell_particle_offset;

                let momentum = weighted_mass * ((*particle).pos_displacement + affine_displacement) / uniforms.simulationTimestep;

                atomicAdd(&grid_momentum_x[cell_index], encodeFixedPoint(momentum.x, uniforms.fixedPointScale));
                atomicAdd(&grid_momentum_y[cell_index], encodeFixedPoint(momentum.y, uniforms.fixedPointScale));
                atomicAdd(&grid_momentum_z[cell_index], encodeFixedPoint(momentum.z, uniforms.fixedPointScale));
                atomicAdd(&grid_mass[cell_index], encodeFixedPoint(weighted_mass, uniforms.fixedPointScale));
            }
        }
    }
}


fn updateGrid(particle: ptr<function, ParticleData>) {
    let start_cell_number = calculateCellNumber((*particle).pos);
    let cell_frac_pos = calculateFractionalPosFromCellMin((*particle).pos, start_cell_number);
    let cell_weights = calculateQuadraticBSplineCellWeights(cell_frac_pos);
    
    // Iterate over 3x3x3 neighborhood
    for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
        for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
            for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                let cell_number = start_cell_number + vec3i(offsetX, offsetY, offsetZ);
                if !cellNumberInGridRange(cell_number) { continue; }

                // O(1) lookup - no hash map!
                let cell_index = cellToGridIndex(cell_number);
                if cell_index == 0xFFFFFFFFu { continue; }
                
                let cell_weight = cell_weights[u32(offsetX + 1)].x
                    * cell_weights[u32(offsetY + 1)].y
                    * cell_weights[u32(offsetZ + 1)].z;

                let weighted_mass = cell_weight * (*particle).mass;

                // Forces: Gravity
                var forces = vec3f(0.0, 0.0, -9.81);

                // Forces: Interaction
                if (uniforms.isInteracting != 0u) {
                    let grid_node_pos = vec3f(cell_number);
                    
                    // Sphere Distance
                    let dist = distance(grid_node_pos, uniforms.interactionPos);

                    if (dist < uniforms.interactionRadius) {
                        let offset = grid_node_pos - uniforms.interactionPos;
                        var push_dir = vec3f(0.0, 0.0, 1.0);
                        if (length(offset) > 0.001) {
                            push_dir = normalize(offset);
                        }
                        
                        var applied_force = vec3f(0.0);

                        let falloff_linear = dist / uniforms.interactionRadius;

                        // 0: Repel
                        if (uniforms.interactionMode == 0u) {
                            let falloff = 1 - falloff_linear * falloff_linear;
                            applied_force = push_dir * uniforms.interactionStrength * falloff;
                        } 
                        // 1: Attract
                        else {
                            let falloff = 1 - falloff_linear * falloff_linear;
                            applied_force = -push_dir * uniforms.interactionStrength * falloff;
                        }

                        forces += applied_force;
                    }
                }

                // PBMPM Loop runs 3 times, so apply 1/3 of the force per iteration
                forces = forces / 3.0;

                let momentum_change = forces * weighted_mass * uniforms.simulationTimestep;

                atomicAdd(&grid_momentum_x[cell_index], encodeFixedPoint(momentum_change.x, uniforms.fixedPointScale));
                atomicAdd(&grid_momentum_y[cell_index], encodeFixedPoint(momentum_change.y, uniforms.fixedPointScale));
                atomicAdd(&grid_momentum_z[cell_index], encodeFixedPoint(momentum_change.z, uniforms.fixedPointScale));
            }
        }
    }
}




@compute
@workgroup_size(256)
fn pbmpm(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let thread_index = gid.x;
    if thread_index >= arrayLength(&particle_data) { return; }

    let particle_index = sortedParticleIndices[thread_index];
    var particle = particle_data[particle_index];

    solveParticleConstraints(&particle);
    transferParticlesToGrid(&particle);
    updateGrid(&particle);

    particle_data[particle_index] = particle;
}
