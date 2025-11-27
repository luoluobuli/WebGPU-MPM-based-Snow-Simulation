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

    if any(pos < uniforms.gridMinCoords) || any(pos >= uniforms.gridMaxCoords) {
        return;
    }

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

    for (var z = 0i; z < 2; z++) {
        for (var y = 0i; y < 2; y++) {
            for (var x = 0i; x < 2; x++) {
                let neighborIndex = vec3i(baseIndex) + vec3i(x, y, z);

                if any(neighborIndex < vec3i(0)) || any(neighborIndex >= vec3i(uniforms.gridResolution)) {
                    continue;
                }

                let weight = 
                    select(w.x, 1 - w.x, x == 0) *
                    select(w.y, 1 - w.y, y == 0) *
                    select(w.z, 1 - w.z, z == 0);

                let idx = linearizeCellIndex(neighborIndex);

                let valToAdd = u32(particle.mass * weight * DENSITY_SCALE);
                atomicAdd(&densityGrid[idx], valToAdd);
            }
        }
    }
}
