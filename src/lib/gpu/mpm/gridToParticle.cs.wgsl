@group(1) @binding(0) var<storage, read_write> hash_map_entries: array<HashMapEntry>;
@group(1) @binding(3) var<storage, read_write> grid_mass: array<atomic<i32>>;
@group(1) @binding(4) var<storage, read_write> grid_momentum_x: array<atomic<i32>>;
@group(1) @binding(5) var<storage, read_write> grid_momentum_y: array<atomic<i32>>;
@group(1) @binding(6) var<storage, read_write> grid_momentum_z: array<atomic<i32>>;

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
    let cellFracPos = calculateFractionalPosFromCellMin(particle.pos, cellDims, startCellNumber);
    let cellWeights = calculateQuadraticBSplineCellWeights(cellFracPos);
    let cellWeightsDeriv = calculateQuadraticBSplineCellWeightDerivatives(cellFracPos);

    // enumerate the 3x3 neighborhood of cells around the cell that contains the particle
    var newParticleVelocity = vec3f(0); 
    var totalVelocityGradient = mat3x3f();
    for (var offsetZ = -1i; offsetZ <= 1i; offsetZ++) {
        for (var offsetY = -1i; offsetY <= 1i; offsetY++) {
            for (var offsetX = -1i; offsetX <= 1i; offsetX++) {
                let cell_number = startCellNumber + vec3i(offsetX, offsetY, offsetZ);
                if !cellNumberInGridRange(cell_number) { continue; }

                let block_number = calculateBlockNumberContainingCell(cell_number);
                let block_index = retrieveBlockIndexFromHashMap(block_number);
                if block_index == GRID_HASH_MAP_BLOCK_INDEX_EMPTY { continue; }

                let cell_index_within_block = calculateCellIndexWithinBlock(cell_number);
                let cell_index = block_index * 64u + cell_index_within_block;
                
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

    if uniforms.use_pbmpm == 0 {
        particle.vel = newParticleVelocity;
        particle.pos += newParticleVelocity * uniforms.simulationTimestep;
        particle.deformationElastic += totalVelocityGradient * particle.deformationElastic * uniforms.simulationTimestep;

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
        particle_data[threadIndex].pos_displacement = newParticleVelocity * uniforms.simulationTimestep;
    }
}