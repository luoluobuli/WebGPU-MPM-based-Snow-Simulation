// Reset indirect buffers for Marching Cubes
// Ensures atomic counters are zeroed out before the frame starts

struct IndirectDispatchArgs {
    x: atomic<u32>,
    y: u32,
    z: u32,
}

struct IndirectDrawArgs {
    vertexCount: atomic<u32>,
    instanceCount: u32,
    firstVertex: u32,
    firstInstance: u32,
}

@group(0) @binding(0) var<storage, read_write> indirectDispatch: IndirectDispatchArgs;
@group(0) @binding(1) var<storage, read_write> indirectDraw: IndirectDrawArgs;

@compute
@workgroup_size(1)
fn main() {
    // Reset block dispatch count
    atomicStore(&indirectDispatch.x, 0u);
    indirectDispatch.y = 1u;
    indirectDispatch.z = 1u;

    // Reset vertex count
    atomicStore(&indirectDraw.vertexCount, 0u);
    indirectDraw.instanceCount = 1u;
    indirectDraw.firstVertex = 0u;
    indirectDraw.firstInstance = 0u;
}
