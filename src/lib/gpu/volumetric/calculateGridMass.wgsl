@group(1) @binding(0) var<storage, read> particleData: array<ParticleData>;
@group(1) @binding(1) var<storage, read_write> mass_grid: array<atomic<u32>>;

@compute
@workgroup_size(256)
fn calculateGridMass(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particleIndex = global_id.x;
    if (particleIndex >= arrayLength(&particleData)) {
        return;
    }

    let particle = particleData[particleIndex];
    let pos = particle.pos;

    if any(pos < uniforms.gridMinCoords) || any(pos >= uniforms.gridMaxCoords) {
        return;
    }

    let gridRange = uniforms.gridMaxCoords - uniforms.gridMinCoords;
    let gridRes = vec3f(uniforms.gridResolution);
    let cellSize = gridRange / gridRes;
    
    let pos_from_grid_min = pos - uniforms.gridMinCoords;
    let pos_cell = pos_from_grid_min / cellSize;
    
    let pos_cell_center = pos_cell - 0.5;
    let start_cell_number = vec3u(pos_cell_center);
    let fractional_pos = pos_cell_center - vec3f(start_cell_number);
    let weights = linearSplineWeights(fractional_pos);

    for (var z = 0u; z < 2; z++) {
        for (var y = 0u; y < 2; y++) {
            for (var x = 0u; x < 2; x++) {
                let cell_number = start_cell_number + vec3u(x, y, z);

                if any(cell_number < vec3u(0)) || any(cell_number >= uniforms.gridResolution) {
                    continue;
                }
                
                let weight = weights[x].x * weights[y].y * weights[z].z;

                let cell_index = linearizeCellIndex(cell_number);

                let mass_contribution = u32(particle.mass * weight * MASS_FIXED_POINT_SCALE);
                atomicAdd(&mass_grid[cell_index], mass_contribution);
            }
        }
    }
}
