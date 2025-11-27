@group(1) @binding(0) var<storage, read> particleData: array<ParticleData>;
@group(1) @binding(1) var<storage, read_write> densityGrid: array<atomic<u32>>;

// Constants
const DENSITY_SCALE = 1000.; // Scale factor for fixed-point storage

@compute @workgroup_size(256)
fn calculateGridDensity(@builtin(global_invocation_id) global_id: vec3<u32>) {
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

    // Trilinear Splatting
    // We splat to the 8 nearest voxel centers.
    // gridPos is in range [0, Res]. 
    // The voxel center for index i is at i + 0.5.
    // So we want to find the 8 voxels surrounding gridPos - 0.5.
    
    let pos_cell_center = pos_cell - 0.5;
    let start_cell_number_i = vec3i(floor(pos_cell_center));
    let fractional_pos = pos_cell_center - vec3f(start_cell_number_i);

    for (var z = 0; z < 2; z++) {
        for (var y = 0; y < 2; y++) {
            for (var x = 0; x < 2; x++) {
                let cell_number_i = start_cell_number_i + vec3i(x, y, z);

                if any(cell_number_i < vec3i(0)) || any(cell_number_i >= vec3i(uniforms.gridResolution)) {
                    continue;
                }
                
                let cell_number = vec3u(cell_number_i);

                let weight = 
                    select(fractional_pos.x, 1 - fractional_pos.x, x == 0) *
                    select(fractional_pos.y, 1 - fractional_pos.y, y == 0) *
                    select(fractional_pos.z, 1 - fractional_pos.z, z == 0);

                let cell_index = linearizeCellIndex(cell_number);

                let mass_contribution = u32(particle.mass * weight * DENSITY_SCALE);
                atomicAdd(&densityGrid[cell_index], mass_contribution);
            }
        }
    }
}
