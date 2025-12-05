// Single-pass marching cubes mesh generation
// One thread per cell, outputs triangles with indirect draw

struct MCParams {
    mcGridRes: vec3u,
    downsampleFactor: u32,
}

struct MCVertex {
    position: vec3f,
    normal: vec3f,
}

struct IndirectDrawArgs {
    vertexCount: atomic<u32>,
    instanceCount: u32,
    firstVertex: u32,
    firstInstance: u32,
}

@group(1) @binding(0) var<storage, read> vertexDensity: array<f32>;
@group(1) @binding(1) var<storage, read> vertexGradient: array<vec4f>;
@group(1) @binding(2) var<storage, read_write> outputVertices: array<MCVertex>;
@group(1) @binding(3) var<storage, read_write> indirectDraw: IndirectDrawArgs;
@group(1) @binding(4) var<uniform> mcParams: MCParams;

const DENSITY_SCALE = 65536.0; // Same as uniforms.fixedPointScale
const ISOVALUE = 0.001; // Low threshold for sparse snow

fn vertexIndex(coord: vec3i) -> u32 {
    let res = vec3i(mcParams.mcGridRes) + vec3i(1);
    return u32(coord.x + coord.y * res.x + coord.z * res.x * res.y);
}

fn getCellVertexDensity(cellCoord: vec3i, vertexOffset: u32) -> f32 {
    // Vertex offset to coord offset
    let dx = i32(vertexOffset & 1u);
    let dy = i32((vertexOffset >> 2u) & 1u);
    let dz = i32((vertexOffset >> 1u) & 1u);
    let vertCoord = cellCoord + vec3i(dx, dy, dz);
    return vertexDensity[vertexIndex(vertCoord)];
}

fn getCellVertexGradient(cellCoord: vec3i, vertexOffset: u32) -> vec3f {
    let dx = i32(vertexOffset & 1u);
    let dy = i32((vertexOffset >> 2u) & 1u);
    let dz = i32((vertexOffset >> 1u) & 1u);
    let vertCoord = cellCoord + vec3i(dx, dy, dz);
    return vertexGradient[vertexIndex(vertCoord)].xyz;
}

fn cellToWorld(cellCoord: vec3f) -> vec3f {
    let gridRange = uniforms.gridMaxCoords - uniforms.gridMinCoords;
    let gridRes = vec3f(mcParams.mcGridRes);
    let cellSize = gridRange / gridRes;
    return uniforms.gridMinCoords + cellCoord * cellSize;
}

@compute
@workgroup_size(8, 8, 4)
fn generateMesh(@builtin(global_invocation_id) global_id: vec3u) {
    let cellCoord = vec3i(global_id);
    let gridRes = vec3i(mcParams.mcGridRes);
    
    if any(cellCoord >= gridRes) {
        return;
    }
    
    // Sample 8 corner densities
    var densities: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        densities[i] = getCellVertexDensity(cellCoord, i);
    }
    
    // Compute marching cubes case
    let mcCase = mcComputeCase(densities, ISOVALUE);
    let numTris = MC_TRI_COUNTS[mcCase];
    
    if numTris == 0u {
        return;
    }
    
    // Allocate vertices atomically
    let baseVertexIdx = atomicAdd(&indirectDraw.vertexCount, numTris * 3u);
    
    // Safety check to prevent buffer overflow (matches MAX_VERTICES in GpuMarchingCubesBufferManager.ts)
    const MAX_VERTICES = 1500000u;
    if (baseVertexIdx + numTris * 3u > MAX_VERTICES) {
        return;
    }
    
    // Generate triangles
    for (var t = 0u; t < numTris; t++) {
        for (var v = 0u; v < 3u; v++) {
            let edgeIdx = mcGetTriangleEdge(mcCase, t, v);
            if edgeIdx < 0 {
                continue;
            }
            
            let edge = u32(edgeIdx);
            
            // Get edge endpoints
            let v0 = MC_EDGE_V0[edge];
            let v1 = MC_EDGE_V1[edge];
            let d0 = densities[v0];
            let d1 = densities[v1];
            
            // Interpolation factor
            let t_interp = clamp((ISOVALUE - d0) / (d1 - d0 + 0.0001), 0.0, 1.0);
            
            // Interpolate position
            let p0 = mcVertexOffset(v0);
            let p1 = mcVertexOffset(v1);
            let localPos = mix(p0, p1, t_interp);
            let worldPos = cellToWorld(vec3f(cellCoord) + localPos);
            
            // Compute analytic gradient of trilinear interpolant at localPos
            // This is robust even if the density field is locally flat at corners (step function)
            // Gradient X
            let dx00 = densities[1] - densities[0];
            let dx10 = densities[5] - densities[4];
            let dx01 = densities[3] - densities[2];
            let dx11 = densities[7] - densities[6];
            let gx = mix(mix(dx00, dx10, localPos.y), mix(dx01, dx11, localPos.y), localPos.z);
            
            // Gradient Y
            let dy00 = densities[4] - densities[0];
            let dy10 = densities[5] - densities[1];
            let dy01 = densities[6] - densities[2];
            let dy11 = densities[7] - densities[3];
            let gy = mix(mix(dy00, dy10, localPos.x), mix(dy01, dy11, localPos.x), localPos.z);
            
            // Gradient Z
            let dz00 = densities[2] - densities[0];
            let dz10 = densities[3] - densities[1];
            let dz01 = densities[6] - densities[4];
            let dz11 = densities[7] - densities[5];
            let gz = mix(mix(dz00, dz10, localPos.x), mix(dz01, dz11, localPos.x), localPos.y);
            
            // Normal points towards lower density
            let normal = -normalize(vec3f(gx, gy, gz));
            
            let vertIdx = baseVertexIdx + t * 3u + v;
            outputVertices[vertIdx].position = worldPos;
            outputVertices[vertIdx].normal = normal;
        }
    }
}
