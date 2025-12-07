// Compute per-vertex density and gradients from the cell density grid
// Sparse version: Dispatched indirectly per active block
// Block size: 8x8x8 cells.
// We compute 9x9x9 vertices to cover the block.
// To compute vertex density, we need density of 8 adjacent cells.
// So for vertices 0..8, we need cells -1..8 (10 cells range).

struct MCParams {
    mcGridRes: vec3u,
    downsampleFactor: u32,
    densityGridRes: vec3u,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> densityGrid: array<u32>;
@group(1) @binding(1) var<storage, read_write> vertexDensity: array<f32>;
@group(1) @binding(2) var<storage, read_write> vertexGradient: array<u32>;
@group(1) @binding(3) var<uniform> mcParams: MCParams;
@group(1) @binding(4) var<storage, read> activeBlocks: array<u32>;

const DENSITY_SCALE = 65536.0;
const BLOCK_SIZE = 8u;

// Get density from a cell in the DENSITY grid (at densityGridRes)
fn getDensityGridValue(cellCoord: vec3i) -> f32 {
    let res = vec3i(mcParams.densityGridRes);
    if (any(cellCoord < vec3i(0)) || any(cellCoord >= res)) {
        return 0.0;
    }
    let idx = cellCoord.x + cellCoord.y * res.x + cellCoord.z * res.x * res.y;
    return f32(densityGrid[idx]) / DENSITY_SCALE;
}

// Sample density at a world position using trilinear interpolation from density grid
fn sampleDensityAtWorldPos(worldPos: vec3f) -> f32 {
    let gridRange = uniforms.gridMaxCoords - uniforms.gridMinCoords;
    let densityRes = vec3f(mcParams.densityGridRes);
    let densityCellSize = gridRange / densityRes;
    
    // Convert world pos to density grid cell coordinates
    let posFromMin = worldPos - uniforms.gridMinCoords;
    let cellPosF = posFromMin / densityCellSize - 0.5;
    let cellPos0 = vec3i(floor(cellPosF));
    let frac = cellPosF - vec3f(cellPos0);
    
    // Trilinear interpolation
    var density = 0.0;
    for (var dz = 0; dz <= 1; dz++) {
        for (var dy = 0; dy <= 1; dy++) {
            for (var dx = 0; dx <= 1; dx++) {
                let cellCoord = cellPos0 + vec3i(dx, dy, dz);
                let wx = select(1.0 - frac.x, frac.x, dx == 1);
                let wy = select(1.0 - frac.y, frac.y, dy == 1);
                let wz = select(1.0 - frac.z, frac.z, dz == 1);
                let weight = wx * wy * wz;
                density += getDensityGridValue(cellCoord) * weight;
            }
        }
    }
    return density;
}

// Get MC vertex world position
fn mcVertexToWorld(vertCoord: vec3i) -> vec3f {
    let gridRange = uniforms.gridMaxCoords - uniforms.gridMinCoords;
    let mcRes = vec3f(mcParams.mcGridRes);
    let mcCellSize = gridRange / mcRes;
    return uniforms.gridMinCoords + vec3f(vertCoord) * mcCellSize;
}

fn vertexIndex(coord: vec3i) -> u32 {
    let res = vec3i(mcParams.mcGridRes) + vec3i(1);
    return u32(coord.x + coord.y * res.x + coord.z * res.x * res.y);
}

@compute
@workgroup_size(8, 8, 4)
fn computeVertexDensity(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(workgroup_id) group_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32
) {
    // 1. Identify which block we are processing
    let blockIdx = activeBlocks[group_id.x];
    
    // Reconstruct 3D block coordinate
    let gridRes = mcParams.mcGridRes;
    let blocksPerAxis = (gridRes + vec3u(BLOCK_SIZE - 1u)) / BLOCK_SIZE;
    let bz = blockIdx / (blocksPerAxis.x * blocksPerAxis.y);
    let rem = blockIdx % (blocksPerAxis.x * blocksPerAxis.y);
    let by = rem / blocksPerAxis.x;
    let bx = rem % blocksPerAxis.x;
    let blockCoord = vec3i(i32(bx), i32(by), i32(bz));
    
    // Base cell for this block (top-left-front)
    let baseCell = blockCoord * 8; // BLOCK_SIZE=8
    
    // No need for shared memory loading anymore - we sample directly
    
    let num_vertices = 729u; // 9^3
    
    for (var i = 0u; i < 3u; i++) {
        let v_idx = local_idx + i * 256u;
        if (v_idx < num_vertices) {
            let lz = i32(v_idx / 81u);
            let rem2 = i32(v_idx % 81u);
            let ly = rem2 / 9;
            let lx = rem2 % 9;
            
            // Current vertex coord in MC grid
            let vertCoord = baseCell + vec3i(lx, ly, lz);
            let vertRes = vec3i(mcParams.mcGridRes) + vec3i(1);
            
            if (all(vertCoord < vertRes) && all(vertCoord >= vec3i(0))) {
                // Determine world position of this vertex
                let worldPos = mcVertexToWorld(vertCoord);
                
                // Sample density at this vertex position
                let density = sampleDensityAtWorldPos(worldPos);
                
                let globalIdx = vertexIndex(vertCoord);
                vertexDensity[globalIdx] = density;
                
                // Compute gradient using finite differences in world space
                // Using a small epsilon related to density grid size
                let gridRange = uniforms.gridMaxCoords - uniforms.gridMinCoords;
                let densityRes = vec3f(mcParams.densityGridRes);
                let densityCellSize = gridRange / densityRes;
                let eps = densityCellSize * 0.5;
                
                let dx = sampleDensityAtWorldPos(worldPos + vec3f(eps.x, 0.0, 0.0)) - 
                         sampleDensityAtWorldPos(worldPos - vec3f(eps.x, 0.0, 0.0));
                let dy = sampleDensityAtWorldPos(worldPos + vec3f(0.0, eps.y, 0.0)) - 
                         sampleDensityAtWorldPos(worldPos - vec3f(0.0, eps.y, 0.0));
                let dz = sampleDensityAtWorldPos(worldPos + vec3f(0.0, 0.0, eps.z)) - 
                         sampleDensityAtWorldPos(worldPos - vec3f(0.0, 0.0, eps.z));
                         
                // Normalize gradient? Original code divided by count/divisor which was sort of normalizing
                // Here we just store the gradient vector. It doesn't need to be normalized yet, mesh gen will normalize.
                // But we should scale it to be somewhat consistent with original scale
                // Original used simple sum differences. Here we use actual differences.
                // Let's just store the vector.
                
                // Normalize and pack gradient
                let grad = vec4f(dx, dy, dz, 0.0);
                var packedGrad = 0u;
                if (length(grad.xyz) > 1e-6) {
                    packedGrad = pack4x8snorm(normalize(grad));
                } else {
                    packedGrad = 0u;
                }
                
                vertexGradient[globalIdx] = packedGrad;
            }
        }
    }
}
