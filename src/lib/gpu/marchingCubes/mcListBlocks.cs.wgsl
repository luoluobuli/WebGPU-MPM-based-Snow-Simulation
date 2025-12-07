// List active blocks for sparse marching cubes
// Each WORKGROUP checks one 8x8x8 block of cells (plus padding)
// We use parallel reduction to check the 10x10x10 neighborhood

struct MCParams {
    mcGridRes: vec3u,
    downsampleFactor: u32,
    densityGridRes: vec3u,
    _padding: u32,
}

struct IndirectDispatchArgs {
    x: atomic<u32>,
    y: u32,
    z: u32,
}

@group(0) @binding(0) var<storage, read> densityGrid: array<u32>;
@group(0) @binding(1) var<storage, read_write> activeBlocks: array<u32>;
@group(0) @binding(2) var<storage, read_write> indirectDispatch: IndirectDispatchArgs;
@group(0) @binding(3) var<uniform> mcParams: MCParams;

const BLOCK_SIZE = 8u;
var<workgroup> s_blockActive: atomic<u32>;

@compute
@workgroup_size(64)
fn listBlocks(
    @builtin(workgroup_id) group_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32
) {
    if (local_idx == 0u) {
        atomicStore(&s_blockActive, 0u);
    }
    workgroupBarrier();

    let blockIdx = group_id.x;
    
    let gridRes = mcParams.mcGridRes;
    let blocksPerAxis = (gridRes + vec3u(BLOCK_SIZE - 1u)) / BLOCK_SIZE;
    let totalBlocks = blocksPerAxis.x * blocksPerAxis.y * blocksPerAxis.z;
    
    if (blockIdx >= totalBlocks) {
        return;
    }
    
    // Calculate block coordinate
    let bz = blockIdx / (blocksPerAxis.x * blocksPerAxis.y);
    let rem = blockIdx % (blocksPerAxis.x * blocksPerAxis.y);
    let by = rem / blocksPerAxis.x;
    let bx = rem % blocksPerAxis.x;
    let blockCoord = vec3i(i32(bx), i32(by), i32(bz));
    
    let startCell = blockCoord * 8;
    
    // We check range -1 to 8 (10x10x10 = 1000 cells)
    // Distributed across 64 threads.
    // Each thread checks ceil(1000/64) = 16 items.
    
    let numItems = 1000u;
    
    for (var i = 0u; i < 16u; i++) {
        let itemIdx = local_idx + i * 64u;
        if (itemIdx < numItems) {
            // Early exit if already active? (Optional optimization, but barrier overhead)
            // Just check.
            
            // Map linear 0..999 to 3D -1..8
            let iz = i32(itemIdx / 100u);
            let rem2 = i32(itemIdx % 100u);
            let iy = rem2 / 10;
            let ix = rem2 % 10;
            
            let offset = vec3i(ix, iy, iz) - vec3i(1); // -1..8
            let cellI = startCell + offset;
            
            // Map MC cell coordinate to density grid coordinate to check for activity
            // MC grid and density grid cover the same world space
            let densityRes = mcParams.densityGridRes;
            let densityCellF = vec3f(cellI) * vec3f(densityRes) / vec3f(gridRes);
            let densityCellI = vec3i(floor(densityCellF));
            
            if (all(densityCellI >= vec3i(0)) && all(densityCellI < vec3i(densityRes))) {
                let cellU = vec3u(densityCellI);
                let idx = cellU.x + cellU.y * densityRes.x + cellU.z * densityRes.x * densityRes.y;
                if (densityGrid[idx] > 0u) {
                    atomicStore(&s_blockActive, 1u);
                }
            }
        }
    }
    
    workgroupBarrier();
    
    if (local_idx == 0u) {
        if (atomicLoad(&s_blockActive) > 0u) {
            let index = atomicAdd(&indirectDispatch.x, 1u);
            activeBlocks[index] = blockIdx;
        }
    }
}
