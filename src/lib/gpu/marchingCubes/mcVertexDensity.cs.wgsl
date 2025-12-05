// Compute per-vertex density and gradients from the cell density grid
// Vertex positions are at cell corners

struct MCParams {
    mcGridRes: vec3u,
    downsampleFactor: u32,
}

@group(1) @binding(0) var<storage, read> densityGrid: array<u32>;
@group(1) @binding(1) var<storage, read_write> vertexDensity: array<f32>;
@group(1) @binding(2) var<storage, read_write> vertexGradient: array<vec4f>;
@group(1) @binding(3) var<uniform> mcParams: MCParams;

const DENSITY_SCALE = 65536.0; // Same as uniforms.fixedPointScale

fn sampleDensity(cellCoord: vec3i) -> f32 {
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

@compute
@workgroup_size(8, 8, 4)
fn computeVertexDensity(@builtin(global_invocation_id) global_id: vec3u) {
    let vertCoord = vec3i(global_id);
    let vertRes = vec3i(mcParams.mcGridRes) + vec3i(1);
    
    if any(vertCoord >= vertRes) {
        return;
    }
    
    // Average density from 8 surrounding cells
    // Also compute gradients by differencing the averages of opposing sides
    var densitySum = 0.0;
    var count = 0.0;
    
    var sumX_pos = 0.0; var sumX_neg = 0.0;
    var sumY_pos = 0.0; var sumY_neg = 0.0;
    var sumZ_pos = 0.0; var sumZ_neg = 0.0;
    
    for (var dz = -1; dz <= 0; dz++) {
        for (var dy = -1; dy <= 0; dy++) {
            for (var dx = -1; dx <= 0; dx++) {
                let cellCoord = vertCoord + vec3i(dx, dy, dz);
                // sampleDensity handles bounds checking internally (returns 0? need to verify)
                // sampleDensity clamps. We want 0 for empty space? 
                // sampleDensity implementation: 
                // let clamped = clamp(cellCoord, vec3i(0), res - vec3i(1));
                // It clamps to edge value. This is good for maintaining solid boundaries.
                
                // For proper gradient at boundary of simulation, we might prefer 0 for out-of-bounds?
                // But for now, let's stick to clamped values.
                
                if all(cellCoord >= vec3i(0)) && all(cellCoord < vec3i(mcParams.mcGridRes)) {
                    let val = sampleDensity(cellCoord);
                    densitySum += val;
                    count += 1.0;
                    
                    if (dx == 0) { sumX_pos += val; } else { sumX_neg += val; }
                    if (dy == 0) { sumY_pos += val; } else { sumY_neg += val; }
                    if (dz == 0) { sumZ_pos += val; } else { sumZ_neg += val; }
                }
            }
        }
    }
    
    if count > 0.0 {
        let idx = vertexIndex(vertCoord);
        vertexDensity[idx] = densitySum / count;
        
        // Compute gradients from the sums
        // approximate avg difference: (sum_pos - sum_neg) / (count / 2)
        let divisor = max(1.0, count * 0.5);
        let gradX = (sumX_pos - sumX_neg) / divisor;
        let gradY = (sumY_pos - sumY_neg) / divisor;
        let gradZ = (sumZ_pos - sumZ_neg) / divisor;
        
        // Normal points towards lower density
        let gradVec = vec3f(gradX, gradY, gradZ);
        let gradLen = length(gradVec);
        var gradient = vec3f(0.0, 1.0, 0.0); // Fallback
        
        if (gradLen > 0.0001) {
            gradient = -gradVec / gradLen;
        }
        
        vertexGradient[idx] = vec4f(gradient, 0.0);
    } else {
        // Should not happen for valid vertices, but safe fallback
        let idx = vertexIndex(vertCoord);
        vertexDensity[idx] = 0.0;
        vertexGradient[idx] = vec4f(0.0, 1.0, 0.0, 0.0);
    }
}
