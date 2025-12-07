struct MCParams {
    mcGridRes: vec3u,
    downsampleFactor: u32,
    densityGridRes: vec3u,
    _padding: u32,
}

struct IndirectDrawArgs {
    vertexCount: atomic<u32>,
    instanceCount: u32,
    firstVertex: u32,
    firstInstance: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read> vertexDensity: array<f32>;
@group(1) @binding(1) var<storage, read> vertexGradient: array<u32>;
@group(1) @binding(2) var<storage, read_write> outputVertices: array<f32>;
@group(1) @binding(3) var<storage, read_write> indirectDraw: IndirectDrawArgs;
@group(1) @binding(4) var<uniform> mcParams: MCParams;
@group(1) @binding(5) var<storage, read> activeBlocks: array<u32>;

const ISOVALUE = 0.08; 
const BLOCK_SIZE = 8u;

fn vertexIndex(coord: vec3i) -> u32 {
    let res = vec3i(mcParams.mcGridRes) + vec3i(1);
    if (any(coord < vec3i(0)) || any(coord >= res)) { return 0u; } 
    return u32(coord.x + coord.y * res.x + coord.z * res.x * res.y);
}

fn cellToWorld(cellCoord: vec3f) -> vec3f {
    let gridRange = uniforms.gridMaxCoords - uniforms.gridMinCoords;
    let gridRes = vec3f(mcParams.mcGridRes);
    let cellSize = gridRange / gridRes;
    return uniforms.gridMinCoords + cellCoord * cellSize;
}

fn getGlobalVertexDensity(vCoord: vec3i) -> f32 {
    let idx = vertexIndex(vCoord);
    return vertexDensity[idx];
}

fn getGlobalVertexGradient(vCoord: vec3i) -> vec3f {
    let idx = vertexIndex(vCoord);
    let packed = vertexGradient[idx];
    let unpacked = unpack4x8snorm(packed);
    return unpacked.xyz;
}

// shared memory for 9x9x9 tile of vertices (8x8x8 + 1 border)
var<workgroup> s_vertexDensity: array<f32, 729>;
var<workgroup> s_vertexGradient: array<vec3f, 729>;

fn sharedIndex(lx: i32, ly: i32, lz: i32) -> i32 {
    return lx + ly * 9 + lz * 81;
}

var<workgroup> wgVertexCount: atomic<u32>;
var<workgroup> wgBaseVertexIdx: u32;

@compute
@workgroup_size(8, 8, 4)
fn generateMesh(
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(workgroup_id) group_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32
) {
    if (local_idx == 0u) {
        atomicStore(&wgVertexCount, 0u);
    }
    
    // Identify block
    let blockIdx = activeBlocks[group_id.x];
    
    let gridRes = mcParams.mcGridRes;
    let blocksPerAxis = (gridRes + vec3u(BLOCK_SIZE - 1u)) / BLOCK_SIZE;
    let bz = blockIdx / (blocksPerAxis.x * blocksPerAxis.y);
    let rem = blockIdx % (blocksPerAxis.x * blocksPerAxis.y);
    let by = rem / blocksPerAxis.x;
    let bx = rem % blocksPerAxis.x;
    let blockCoord = vec3i(i32(bx), i32(by), i32(bz));
    
    let baseCell = blockCoord * 8;
    let num_loads = 729u;
    
    // Cooperative load
    for (var i = 0u; i < 3u; i++) {
        let idx = local_idx + i * 256u;
        if (idx < num_loads) {
            let lz = i32(idx / 81u);
            let rem2 = i32(idx % 81u);
            let ly = rem2 / 9;
            let lx = rem2 % 9;
            
            let vPos = baseCell + vec3i(lx, ly, lz);
            s_vertexDensity[idx] = getGlobalVertexDensity(vPos);
            s_vertexGradient[idx] = getGlobalVertexGradient(vPos);
        }
    }
    
    workgroupBarrier();

    // Loop
    let num_cells = 512u; // 8^3
    
    var savedCases: array<u32, 2>;
    var savedOffsets: array<u32, 2>;
    var savedNumTris: array<u32, 2>;
    
    for (var i = 0u; i < 2u; i++) {
        savedNumTris[i] = 0u;
        
        let cell_idx = local_idx + i * 256u;
        if (cell_idx < num_cells) {
            let cz = i32(cell_idx / 64u);
            let rem2 = i32(cell_idx % 64u);
            let cy = rem2 / 8;
            let cx = rem2 % 8;
            
            let lx = cx; let ly = cy; let lz = cz;
            let cellGlobal = baseCell + vec3i(cx, cy, cz);
            
            if (all(cellGlobal < vec3i(gridRes))) {
                var densities: array<f32, 8>;
                densities[0] = s_vertexDensity[sharedIndex(lx, ly, lz)];
                densities[1] = s_vertexDensity[sharedIndex(lx+1, ly, lz)];
                densities[2] = s_vertexDensity[sharedIndex(lx, ly, lz+1)];
                densities[3] = s_vertexDensity[sharedIndex(lx+1, ly, lz+1)];
                densities[4] = s_vertexDensity[sharedIndex(lx, ly+1, lz)];
                densities[5] = s_vertexDensity[sharedIndex(lx+1, ly+1, lz)];
                densities[6] = s_vertexDensity[sharedIndex(lx, ly+1, lz+1)];
                densities[7] = s_vertexDensity[sharedIndex(lx+1, ly+1, lz+1)];
                
                let mcCase = mcComputeCase(densities, ISOVALUE);
                let numTris = MC_TRI_COUNTS[mcCase];
                
                if (numTris > 0u) {
                    savedCases[i] = mcCase;
                    savedNumTris[i] = numTris;
                    savedOffsets[i] = atomicAdd(&wgVertexCount, numTris * 3u);
                }
            }
        }
    }
    
    workgroupBarrier();
    
    if (local_idx == 0u) {
        let totalWgCount = atomicLoad(&wgVertexCount);
        if (totalWgCount > 0u) {
            wgBaseVertexIdx = atomicAdd(&indirectDraw.vertexCount, totalWgCount);
        }
    }
    
    workgroupBarrier();
    
    let baseIdx = wgBaseVertexIdx;
    let MAX_VERTICES = 10500000u; // Matches buffer (~3.5M tris * 3)
    
    // Output pass
    for (var i = 0u; i < 2u; i++) {
        let numTris = savedNumTris[i];
        if (numTris > 0u) {
            let mcCase = savedCases[i];
            let myOffset = savedOffsets[i];
            
            let cell_idx = local_idx + i * 256u;
            let cz = i32(cell_idx / 64u);
            let rem2 = i32(cell_idx % 64u);
            let cy = rem2 / 8;
            let cx = rem2 % 8;
            let cellGlobal = baseCell + vec3i(cx, cy, cz);
            
            let lx = cx; let ly = cy; let lz = cz;
            
            var densities: array<f32, 8>;
            var gradients: array<vec3f, 8>;
            
            let s0 = sharedIndex(lx, ly, lz);
            let s1 = sharedIndex(lx+1, ly, lz);
            let s2 = sharedIndex(lx, ly, lz+1);
            let s3 = sharedIndex(lx+1, ly, lz+1);
            let s4 = sharedIndex(lx, ly+1, lz);
            let s5 = sharedIndex(lx+1, ly+1, lz);
            let s6 = sharedIndex(lx, ly+1, lz+1);
            let s7 = sharedIndex(lx+1, ly+1, lz+1);
            
            densities[0] = s_vertexDensity[s0]; gradients[0] = s_vertexGradient[s0];
            densities[1] = s_vertexDensity[s1]; gradients[1] = s_vertexGradient[s1];
            densities[2] = s_vertexDensity[s2]; gradients[2] = s_vertexGradient[s2];
            densities[3] = s_vertexDensity[s3]; gradients[3] = s_vertexGradient[s3];
            densities[4] = s_vertexDensity[s4]; gradients[4] = s_vertexGradient[s4];
            densities[5] = s_vertexDensity[s5]; gradients[5] = s_vertexGradient[s5];
            densities[6] = s_vertexDensity[s6]; gradients[6] = s_vertexGradient[s6];
            densities[7] = s_vertexDensity[s7]; gradients[7] = s_vertexGradient[s7];
            
            for (var t = 0u; t < numTris; t++) {
                for (var v = 0u; v < 3u; v++) {
                    let edgeIdx = mcGetTriangleEdge(mcCase, t, v);
                    let edge = u32(edgeIdx);
                    
                    let v0 = MC_EDGE_V0[edge];
                    let v1 = MC_EDGE_V1[edge];
                    let d0 = densities[v0];
                    let d1 = densities[v1];
                    
                    let t_interp = clamp((ISOVALUE - d0) / (d1 - d0 + 0.0001), 0.0, 1.0);
                    
                    let p0 = mcVertexOffset(v0);
                    let p1 = mcVertexOffset(v1);
                    let localPos = mix(p0, p1, t_interp);
                    let worldPos = cellToWorld(vec3f(cellGlobal) + localPos);
                    
                    let g0 = mix(gradients[0], gradients[1], localPos.x); 
                    let g1 = mix(gradients[2], gradients[3], localPos.x); 
                    let g2 = mix(gradients[4], gradients[5], localPos.x); 
                    let g3 = mix(gradients[6], gradients[7], localPos.x); 
                    let gy0 = mix(g0, g1, localPos.z);
                    let gy1 = mix(g2, g3, localPos.z);
                    let normalVec = mix(gy0, gy1, localPos.y);
                    
                    let nLen = length(normalVec);
                    var normal = vec3f(0.0);
                    if (nLen > 0.0001) {
                        normal = -normalVec / nLen;
                    }
                    
                    let globalVertIdx = baseIdx + myOffset + t * 3u + v;
                    if (globalVertIdx < MAX_VERTICES) {
                        // Packed write: 6 floats per vertex
                        let floatIdx = globalVertIdx * 6u;
                        outputVertices[floatIdx + 0u] = worldPos.x;
                        outputVertices[floatIdx + 1u] = worldPos.y;
                        outputVertices[floatIdx + 2u] = worldPos.z;
                        outputVertices[floatIdx + 3u] = normal.x;
                        outputVertices[floatIdx + 4u] = normal.y;
                        outputVertices[floatIdx + 5u] = normal.z;
                    }
                }
            }
        }
    }
}
