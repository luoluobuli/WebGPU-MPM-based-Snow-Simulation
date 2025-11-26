@group(1) @binding(0) var<storage, read> particleData: array<ParticleData>;
@group(1) @binding(1) var<storage, read_write> densityGrid: array<atomic<u32>>;

// Constants
const DENSITY_SCALE: f32 = 1000.0; // Scale factor for fixed-point storage

@compute @workgroup_size(256)
fn doDensityRasterize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particleIndex = global_id.x;
    if (particleIndex >= arrayLength(&particleData)) {
        return;
    }

    let particle = particleData[particleIndex];
    let pos = particle.pos;

    // Check bounds
    if (any(pos < uniforms.gridMinCoords) || any(pos >= uniforms.gridMaxCoords)) {
        return;
    }

    // Map to grid coordinates
    let gridRange = uniforms.gridMaxCoords - uniforms.gridMinCoords;
    let gridRes = vec3f(uniforms.gridResolution);
    let cellSize = gridRange / gridRes;
    
    let localPos = pos - uniforms.gridMinCoords;
    let gridPos = localPos / cellSize;

    // Trilinear Splatting
    // We splat to the 8 nearest voxel centers.
    // gridPos is in range [0, Res]. 
    // The voxel center for index i is at i + 0.5.
    // So we want to find the 8 voxels surrounding gridPos - 0.5.
    
    let splatPos = gridPos - 0.5;
    let baseIndex = floor(splatPos); // The "min" corner voxel index
    let w = fract(splatPos); // Weights for trilinear interpolation

    // Iterate over 2x2x2 neighborhood
    for (var z = 0u; z < 2u; z++) {
        for (var y = 0u; y < 2u; y++) {
            for (var x = 0u; x < 2u; x++) {
                let neighborIndex = vec3f(baseIndex) + vec3f(f32(x), f32(y), f32(z));
                let neighborU32 = vec3u(neighborIndex);

                // Bounds check for neighbor
                if (any(neighborIndex < vec3f(0.0)) || any(neighborIndex >= gridRes)) {
                    continue;
                }

                // Compute weight
                let weight = 
                    select(1.0 - w.x, w.x, x == 0u) *
                    select(1.0 - w.y, w.y, y == 0u) *
                    select(1.0 - w.z, w.z, z == 0u);

                // Linear index
                let idx = neighborU32.x + 
                          neighborU32.y * uniforms.gridResolution.x + 
                          neighborU32.z * uniforms.gridResolution.x * uniforms.gridResolution.y;

                // Atomic Add
                // We use particle mass as the base quantity.
                // Density = Mass / Volume. Here we just accumulate Mass-like quantity.
                // The ray marcher will divide by cell volume if needed, or we just tune the scale.
                let valToAdd = u32(particle.mass * weight * DENSITY_SCALE);
                atomicAdd(&densityGrid[idx], valToAdd);
            }
        }
    }
}
