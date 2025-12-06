// Clamps the indirect draw arguments to the maximum buffer size
// preventing the GPU from attempting to render out-of-bounds (unwritten/stale) vertices

struct IndirectDrawArgs {
    vertexCount: atomic<u32>,
    instanceCount: u32,
    firstVertex: u32,
    firstInstance: u32,
}

struct MCParams {
    mcGridRes: vec3u,
    downsampleFactor: u32,
    maxVertices: u32, // Passed via uniform
}

@group(0) @binding(0) var<storage, read_write> indirectDraw: IndirectDrawArgs;
@group(0) @binding(1) var<uniform> mcParams: MCParams; // Reuse checking params or separate? 
// Actually we can just hardcode or pass a uniform. 
// Reusing standard MCParams might be cleaner if we add maxVertices there.
// For now, let's just use a hardcoded limit matching the buffer manager for simplicity/speed,
// or pass a dedicated uniform.

// Let's use a specialized binding for safety.
@group(0) @binding(2) var<uniform> maxVertsUniform: u32; 

@compute
@workgroup_size(1)
fn main() {
    let count = atomicLoad(&indirectDraw.vertexCount);
    // Don't modify count in place atomically if we want to preserve the "real" count for debug?
    // No, for drawing we MUST clamp it.
    if (count > maxVertsUniform) {
        atomicStore(&indirectDraw.vertexCount, maxVertsUniform);
    }
}
