// Compute per-vertex density and gradients from the cell density grid
// Sparse version: Dispatched indirectly per active block
// Block size: 8x8x8 cells.
// We compute 9x9x9 vertices to cover the block.
// To compute vertex density, we need density of 8 adjacent cells.
// So for vertices 0..8, we need cells -1..8 (10 cells range).

struct MCParams {
    mcGridRes: vec3u,
    downsampleFactor: u32,
}

@group(1) @binding(0) var<storage, read> densityGrid: array<u32>;
@group(1) @binding(1) var<storage, read_write> vertexDensity: array<f32>;
@group(1) @binding(2) var<storage, read_write> vertexGradient: array<vec4f>;
@group(1) @binding(3) var<uniform> mcParams: MCParams;
@group(1) @binding(4) var<storage, read> activeBlocks: array<u32>;

const DENSITY_SCALE = 65536.0;
const BLOCK_SIZE = 8u;

fn getGlobalDensity(cellCoord: vec3i) -> f32 {
    let res = vec3i(mcParams.mcGridRes);
    if (any(cellCoord < vec3i(0)) || any(cellCoord >= res)) {
        return 0.0;
    }
    let idx = cellCoord.x + cellCoord.y * res.x + cellCoord.z * res.x * res.y;
    return f32(densityGrid[idx]) / DENSITY_SCALE;
}

fn vertexIndex(coord: vec3i) -> u32 {
    let res = vec3i(mcParams.mcGridRes) + vec3i(1);
    return u32(coord.x + coord.y * res.x + coord.z * res.x * res.y);
}

// Shared memory for 10x10x10 tile of density values
// Size: 1000 entries
var<workgroup> s_density: array<f32, 1000>;

fn sharedIndex(lx: i32, ly: i32, lz: i32) -> i32 {
    // lx, ly, lz are 0..9
    return lx + ly * 10 + lz * 100;
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
    // Indirect dispatch x = number of active blocks
    // group_id.x is the index into activeBlocks array
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
    
    // We need cells from baseCell - 1 to baseCell + 8
    // This is a 10x10x10 volume starting at baseCell - 1
    let volumeStart = baseCell - vec3i(1);
    
    let num_loads = 1000u;
    
    // Cooperative load
    // 256 threads. Need 4 loops to cover 1000 items.
    for (var i = 0u; i < 4u; i++) {
        let idx = local_idx + i * 256u;
        if (idx < num_loads) {
            let lz = i32(idx / 100u);
            let rem2 = i32(idx % 100u);
            let ly = rem2 / 10;
            let lx = rem2 % 10;
            
            let cellPos = volumeStart + vec3i(lx, ly, lz);
            s_density[idx] = getGlobalDensity(cellPos);
        }
    }
    
    workgroupBarrier();
    
    // Compute for 9x9x9 vertices
    // We have 256 threads (8x8x4).
    // Let's iterate z to cover 0..8
    // Thread covers (lx, ly) 0..7
    // That's 8x8 = 64. 4 z-layers?
    // We need 9x9x9 = 729 vertices.
    // 256 threads * 3 = 768.
    
    // Process items by 3 chunks?
    // Or just map local_idx to 0..728
    let num_vertices = 729u; // 9^3
    
    for (var i = 0u; i < 3u; i++) {
        let v_idx = local_idx + i * 256u;
        if (v_idx < num_vertices) {
            let lz = i32(v_idx / 81u);
            let rem2 = i32(v_idx % 81u);
            let ly = rem2 / 9;
            let lx = rem2 % 9;
            
            // This is the vertex coordinate relative to baseCell
            // Relative to volumeStart (which is -1), offset is lx+1, ly+1, lz+1
            // Actually:
            // vertices are 0..8 relative to baseCell
            // s_density indices 0..9 correspond to cells -1..8
            // vertex at 0 needs cells -1, 0. s_density index 0, 1.
            // vertex at x needs cells x-1, x. s_density index x, x+1.
            
            // So s_idx offsets are just lx, ly, lz (for cell -1) and +1 for cell 0
            
            let vertCoord = baseCell + vec3i(lx, ly, lz);
            let vertRes = vec3i(mcParams.mcGridRes) + vec3i(1);
            
            if (all(vertCoord < vertRes) && all(vertCoord >= vec3i(0))) {
                
                var densitySum = 0.0;
                var count = 0.0;
                var sumX_pos = 0.0; var sumX_neg = 0.0;
                var sumY_pos = 0.0; var sumY_neg = 0.0;
                var sumZ_pos = 0.0; var sumZ_neg = 0.0;
                
                // For a vertex at integer coord V, it is surrounded by cells:
                // (V-1), (V) in 3D.
                // dx, dy, dz in -1..0
                // s_density index corresponds to cell coord.
                // vertCoord + d corresponds to actual cell.
                // local s index: (lx + dx + 1) NO.
                // Let's trace:
                // vertCoord.x corresponds to lx relative to baseCell.
                // cell depends on dx (-1 or 0).
                // cell x = vertCoord.x + dx = baseCell.x + lx + dx.
                // volumeStart.x = baseCell.x - 1.
                // offset in s_density = cell.x - volumeStart.x = lx + dx + 1.
                
                // Example: lx=0, dx=-1. offset = 0 - 1 + 1 = 0. Correct (index 0).
                // Example: lx=0, dx=0. offset = 0 + 0 + 1 = 1. Correct (index 1).
                
                let sx = lx + 1;
                let sy = ly + 1;
                let sz = lz + 1;
                
                for (var dz = -1; dz <= 0; dz++) {
                    for (var dy = -1; dy <= 0; dy++) {
                        for (var dx = -1; dx <= 0; dx++) {
                            let s_idx = sharedIndex(sx + dx, sy + dy, sz + dz);
                            let val = s_density[s_idx];
                            
                            // Check boundary of ORIGINAL cells for 'count'
                            let cellGlobal = vertCoord + vec3i(dx, dy, dz);
                            if (all(cellGlobal >= vec3i(0)) && all(cellGlobal < vec3i(mcParams.mcGridRes))) {
                                densitySum += val;
                                count += 1.0;
                                
                                if (dx == 0) { sumX_pos += val; } else { sumX_neg += val; }
                                if (dy == 0) { sumY_pos += val; } else { sumY_neg += val; }
                                if (dz == 0) { sumZ_pos += val; } else { sumZ_neg += val; }
                            }
                        }
                    }
                }
                
                let globalIdx = vertexIndex(vertCoord);
                
                if (count > 0.0) {
                    vertexDensity[globalIdx] = densitySum / count;
                    let divisor = max(1.0, count * 0.5);
                    vertexGradient[globalIdx] = vec4f(
                        (sumX_pos - sumX_neg) / divisor,
                        (sumY_pos - sumY_neg) / divisor,
                        (sumZ_pos - sumZ_neg) / divisor,
                        0.0
                    );
                } else {
                    vertexDensity[globalIdx] = 0.0;
                    vertexGradient[globalIdx] = vec4f(0.0, 1.0, 0.0, 0.0);
                }
            }
        }
    }
}

