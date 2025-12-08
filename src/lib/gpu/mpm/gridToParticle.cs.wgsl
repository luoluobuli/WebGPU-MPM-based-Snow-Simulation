@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Dense grid bind group (no sparse_grid buffer needed)
@group(1) @binding(3) var<storage, read_write> grid_mass: array<i32>;
@group(1) @binding(4) var<storage, read_write> grid_momentum_x: array<i32>;
@group(1) @binding(5) var<storage, read_write> grid_momentum_y: array<i32>;
@group(1) @binding(6) var<storage, read_write> grid_momentum_z: array<i32>;

@group(2) @binding(0) var<storage, read_write> particle_data: array<ParticleData>;
@group(2) @binding(1) var<storage, read_write> sortedParticleIndices: array<u32>;

@compute
@workgroup_size(256)
fn doGridToParticle(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let thread_index = gid.x;
    if thread_index >= arrayLength(&particle_data) { return; }

    let particle_index = sortedParticleIndices[thread_index];
    var particle = particle_data[particle_index];

    let start_cell_number = calculateCellNumber(particle.pos);
    let cell_center_pos = uniforms.gridMinCoords + uniforms.gridCellDims * (vec3f(start_cell_number) + vec3f(0.5));
    let cell_frac_pos = calculateFractionalPosFromCellMin(particle.pos, start_cell_number);
    let cell_weights = calculateQuadraticBSplineCellWeights(cell_frac_pos);

    if uniforms.use_pbmpm == 0 {
        let cell_weights_deriv = calculateQuadraticBSplineCellWeightDerivatives(cell_frac_pos);

        // enumerate the 3x3 neighborhood of cells around the cell that contains the particle
        var new_particle_velocity = vec3f(0); 
        var total_velocity_gradient = mat3x3f();
        for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
            for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
                for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                    let cell_number = start_cell_number + vec3i(offsetX, offsetY, offsetZ);
                    if !cellNumberInGridRange(cell_number) { continue; }

                    // O(1) direct indexing - no hash map!
                    let cell_index = cellToGridIndex(cell_number);
                    if cell_index == 0xFFFFFFFFu { continue; }
                    
                    let cell_mass = f32(grid_mass[cell_index]) / uniforms.fixedPointScale;
                    if cell_mass <= 0 { continue; }

                    let cell_momentum_x = f32(grid_momentum_x[cell_index]) / uniforms.fixedPointScale;
                    let cell_momentum_y = f32(grid_momentum_y[cell_index]) / uniforms.fixedPointScale;
                    let cell_momentum_z = f32(grid_momentum_z[cell_index]) / uniforms.fixedPointScale;
                    let cell_velocity = vec3f(cell_momentum_x, cell_momentum_y, cell_momentum_z) / cell_mass;

                    
                    let cell_weight = cell_weights[u32(offsetX + 1)].x
                        * cell_weights[u32(offsetY + 1)].y
                        * cell_weights[u32(offsetZ + 1)].z;
                        
                    new_particle_velocity += cell_weight * cell_velocity;

                    let cell_weight_gradient = vec3f(
                        cell_weights_deriv[u32(offsetX + 1)].x * cell_weights[u32(offsetY + 1)].y * cell_weights[u32(offsetZ + 1)].z,
                        cell_weights[u32(offsetX + 1)].x * cell_weights_deriv[u32(offsetY + 1)].y * cell_weights[u32(offsetZ + 1)].z,
                        cell_weights[u32(offsetX + 1)].x * cell_weights[u32(offsetY + 1)].y * cell_weights_deriv[u32(offsetZ + 1)].z,
                    );

                    total_velocity_gradient += mat3x3f(
                        cell_weight_gradient.x * cell_velocity,
                        cell_weight_gradient.y * cell_velocity,
                        cell_weight_gradient.z * cell_velocity,
                    );
                }
            }
        }

        particle.vel = new_particle_velocity;
        
        // Defer position and deformation update to integrateParticles
        particle.pos_displacement = new_particle_velocity * uniforms.simulationTimestep;
        particle.deformation_displacement = total_velocity_gradient * uniforms.simulationTimestep;
        
        particle_data[particle_index] = particle;
    }

    else {
        let particle_cell_pos = vec3f(start_cell_number) + cell_frac_pos - 0.5;

        // enumerate the 3x3 neighborhood of cells around the cell that contains the particle
        var new_particle_velocity = vec3f(0); 
        var B = mat3x3f(
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        );

        for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
            for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
                for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                    let cell_number = start_cell_number + vec3i(offsetX, offsetY, offsetZ);
                    if !cellNumberInGridRange(cell_number) { continue; }

                    // O(1) direct indexing - no hash map!
                    let cell_index = cellToGridIndex(cell_number);
                    if cell_index == 0xFFFFFFFFu { continue; }
                    
                    let cell_mass = f32(grid_mass[cell_index]) / uniforms.fixedPointScale;
                    if cell_mass <= 0 { continue; }

                    let cell_momentum_x = f32(grid_momentum_x[cell_index]) / uniforms.fixedPointScale;
                    let cell_momentum_y = f32(grid_momentum_y[cell_index]) / uniforms.fixedPointScale;
                    let cell_momentum_z = f32(grid_momentum_z[cell_index]) / uniforms.fixedPointScale;
                    let cell_velocity = vec3f(cell_momentum_x, cell_momentum_y, cell_momentum_z) / cell_mass;

                    
                    let cell_weight = cell_weights[u32(offsetX + 1)].x
                        * cell_weights[u32(offsetY + 1)].y
                        * cell_weights[u32(offsetZ + 1)].z;
                        
                    new_particle_velocity += cell_weight * cell_velocity;

                    let dist = vec3f(cell_number) - particle_cell_pos;
                    B += outerProduct(cell_weight * cell_velocity, dist);
                }
            }
        }

        particle.pos_displacement = new_particle_velocity * uniforms.simulationTimestep;
        particle.deformation_displacement = B * uniforms.simulationTimestep * 4.0;

        // temp
        particle.vel = new_particle_velocity;

        particle_data[particle_index] = particle;
    }
}