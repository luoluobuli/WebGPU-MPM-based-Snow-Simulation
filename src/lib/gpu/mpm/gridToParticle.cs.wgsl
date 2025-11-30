@group(1) @binding(0) var<storage, read_write> hash_map_entries: array<HashMapEntry>;
@group(1) @binding(3) var<storage, read_write> grid_mass: array<atomic<i32>>;
@group(1) @binding(4) var<storage, read_write> grid_momentum_x: array<atomic<i32>>;
@group(1) @binding(5) var<storage, read_write> grid_momentum_y: array<atomic<i32>>;
@group(1) @binding(6) var<storage, read_write> grid_momentum_z: array<atomic<i32>>;
@group(1) @binding(7) var<storage, read_write> grid_mass_displacement_x: array<atomic<i32>>;
@group(1) @binding(8) var<storage, read_write> grid_mass_displacement_y: array<atomic<i32>>;
@group(1) @binding(9) var<storage, read_write> grid_mass_displacement_z: array<atomic<i32>>;

@group(2) @binding(0) var<storage, read_write> particle_data: array<ParticleData>;

@compute
@workgroup_size(256)
fn doGridToParticle(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let threadIndex = gid.x;
    if threadIndex >= arrayLength(&particle_data) { return; }

    var particle = particle_data[threadIndex];

    let cellDims = calculateCellDims();
    let startCellNumber = calculateCellNumber(particle.pos, cellDims);
    let cell_center_pos = uniforms.gridMinCoords + cellDims * (vec3f(startCellNumber) + vec3f(0.5));
    let cellFracPos = calculateFractionalPosFromCellMin(particle.pos, cellDims, startCellNumber);
    let cellWeights = calculateQuadraticBSplineCellWeights(cellFracPos);
    let cellWeightsDeriv = calculateQuadraticBSplineCellWeightDerivatives(cellFracPos);

    if uniforms.use_pbmpm == 0 {
        // enumerate the 3x3 neighborhood of cells around the cell that contains the particle
        var newParticleVelocity = vec3f(0); 
        var totalVelocityGradient = mat3x3f();
        for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
            for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
                for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                    let cell_number = startCellNumber + vec3i(offsetX, offsetY, offsetZ);
                    if !cellNumberInGridRange(cell_number) { continue; }

                    let cell_index = calculateCellIndexFromCellNumber(cell_number);
                    if cell_index == GRID_HASH_MAP_BLOCK_INDEX_EMPTY { continue; }
                    
                    let cellMass = f32(atomicLoad(&grid_mass[cell_index])) / uniforms.fixedPointScale;
                    if cellMass <= 0 { continue; }

                    let cellMomentumX = f32(atomicLoad(&grid_momentum_x[cell_index])) / uniforms.fixedPointScale;
                    let cellMomentumY = f32(atomicLoad(&grid_momentum_y[cell_index])) / uniforms.fixedPointScale;
                    let cellMomentumZ = f32(atomicLoad(&grid_momentum_z[cell_index])) / uniforms.fixedPointScale;
                    let cellVelocity = vec3f(cellMomentumX, cellMomentumY, cellMomentumZ) / cellMass;

                    
                    let cellWeight = cellWeights[u32(offsetX + 1)].x
                        * cellWeights[u32(offsetY + 1)].y
                        * cellWeights[u32(offsetZ + 1)].z;
                        
                    newParticleVelocity += cellWeight * cellVelocity;

                    let cellWeightGradient = vec3f(
                        cellWeightsDeriv[u32(offsetX + 1)].x * cellWeights[u32(offsetY + 1)].y * cellWeights[u32(offsetZ + 1)].z,
                        cellWeights[u32(offsetX + 1)].x * cellWeightsDeriv[u32(offsetY + 1)].y * cellWeights[u32(offsetZ + 1)].z,
                        cellWeights[u32(offsetX + 1)].x * cellWeights[u32(offsetY + 1)].y * cellWeightsDeriv[u32(offsetZ + 1)].z,
                    );

                    totalVelocityGradient += mat3x3f(
                        cellWeightGradient.x * cellVelocity,
                        cellWeightGradient.y * cellVelocity,
                        cellWeightGradient.z * cellVelocity,
                    );
                }
            }
        }

        particle.vel = newParticleVelocity;
        particle.pos += newParticleVelocity * uniforms.simulationTimestep;
        particle.deformationElastic = (mat3x3Identity() + totalVelocityGradient * uniforms.simulationTimestep) * particle.deformationElastic;

        applyPlasticity(&particle);
        
        // Boundary conditions
        if particle.pos.x < uniforms.gridMinCoords.x {
            particle.vel.x *= -0.5;
            particle.pos.x = uniforms.gridMinCoords.x;
        }
        if particle.pos.x >= uniforms.gridMaxCoords.x {
            particle.vel.x *= -0.5;
            particle.pos.x = uniforms.gridMaxCoords.x;
        }

        if particle.pos.y < uniforms.gridMinCoords.y {
            particle.vel.y *= -0.5;
            particle.pos.y = uniforms.gridMinCoords.y;
        }
        if particle.pos.y >= uniforms.gridMaxCoords.y {
            particle.vel.y *= -0.5;
            particle.pos.y = uniforms.gridMaxCoords.y;
        }

        if particle.pos.z < uniforms.gridMinCoords.z {
            particle.vel.z *= -0.5;
            particle.pos.z = uniforms.gridMinCoords.z;
        }
        if particle.pos.z >= uniforms.gridMaxCoords.z {
            particle.vel.z *= -0.5;
            particle.pos.z = uniforms.gridMaxCoords.z;
        }

        particle_data[threadIndex] = particle;
    }

    else {
        // enumerate the 3x3 neighborhood of cells around the cell that contains the particle
        var new_particle_displacement = vec3f(0); 
        var total_displacement_gradient = mat3x3f();
        for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
            for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
                for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                    let cell_number = startCellNumber + vec3i(offsetX, offsetY, offsetZ);
                    if !cellNumberInGridRange(cell_number) { continue; }

                    let cell_index = calculateCellIndexFromCellNumber(cell_number);
                    if cell_index == GRID_HASH_MAP_BLOCK_INDEX_EMPTY { continue; }
                    
                    let cellMass = f32(atomicLoad(&grid_mass[cell_index])) / uniforms.fixedPointScale;
                    if cellMass <= 0 { continue; }

                    let cell_mass_displacement_x = f32(atomicLoad(&grid_mass_displacement_x[cell_index])) / uniforms.fixedPointScale;
                    let cell_mass_displacement_y = f32(atomicLoad(&grid_mass_displacement_y[cell_index])) / uniforms.fixedPointScale;
                    let cell_mass_displacement_z = f32(atomicLoad(&grid_mass_displacement_z[cell_index])) / uniforms.fixedPointScale;
                    let cell_displacement = vec3f(cell_mass_displacement_x, cell_mass_displacement_y, cell_mass_displacement_z) / cellMass;

                    
                    let cellWeight = cellWeights[u32(offsetX + 1)].x
                        * cellWeights[u32(offsetY + 1)].y
                        * cellWeights[u32(offsetZ + 1)].z;
                        
                    new_particle_displacement += cellWeight * cell_displacement;

                    let cellWeightGradient = vec3f(
                        cellWeightsDeriv[u32(offsetX + 1)].x * cellWeights[u32(offsetY + 1)].y * cellWeights[u32(offsetZ + 1)].z,
                        cellWeights[u32(offsetX + 1)].x * cellWeightsDeriv[u32(offsetY + 1)].y * cellWeights[u32(offsetZ + 1)].z,
                        cellWeights[u32(offsetX + 1)].x * cellWeights[u32(offsetY + 1)].y * cellWeightsDeriv[u32(offsetZ + 1)].z,
                    );

                    total_displacement_gradient += mat3x3f(
                        cellWeightGradient.x * cell_displacement,
                        cellWeightGradient.y * cell_displacement,
                        cellWeightGradient.z * cell_displacement,
                    );
                }
            }
        }

        particle.pos_displacement = new_particle_displacement;
        particle.deformation_displacement = total_displacement_gradient;

        particle_data[threadIndex] = particle;
    }
}