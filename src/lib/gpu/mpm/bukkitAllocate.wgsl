// bukkitAllocate.wgsl - Allocate thread data and particle ranges for each bukkit

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> bukkitParams: BukkitParams;

@group(1) @binding(0) var<storage, read_write> bukkitCounts: array<u32>;
@group(1) @binding(1) var<storage, read_write> bukkitDispatch: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> bukkitThreadData: array<BukkitThreadData>;
// Bindings 3-6 skipped (grid mass/momentum)
@group(1) @binding(7) var<storage, read_write> bukkitParticleAllocator: atomic<u32>;
@group(1) @binding(8) var<storage, read_write> bukkitIndexStart: array<u32>;

fn divUp(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

@compute @workgroup_size(8, 8, 4)
fn bukkitAllocate(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= bukkitParams.countX || 
        gid.y >= bukkitParams.countY ||
        gid.z >= bukkitParams.countZ) {
        return;
    }
    
    let bukkitIndex = bukkitIdToIndex(gid, bukkitParams.countX, bukkitParams.countY);
    let count = bukkitCounts[bukkitIndex];
    
    if (count == 0u) { return; }
    
    let dispatchCount = divUp(count, PARTICLE_DISPATCH_SIZE);
    let dispatchStartIndex = atomicAdd(&bukkitDispatch[0], dispatchCount);
    let particleStartIndex = atomicAdd(&bukkitParticleAllocator, count);
    
    bukkitIndexStart[bukkitIndex] = particleStartIndex;
    
    let countResidual = count % PARTICLE_DISPATCH_SIZE;
    
    for (var i = 0u; i < dispatchCount; i++) {
        var groupCount = PARTICLE_DISPATCH_SIZE;
        if (countResidual != 0u && i == dispatchCount - 1u) {
            groupCount = countResidual;
        }
        
        bukkitThreadData[i + dispatchStartIndex] = BukkitThreadData(
            particleStartIndex + i * PARTICLE_DISPATCH_SIZE,
            groupCount,
            gid.x,
            gid.y,
            gid.z,
            0u
        );
    }
}
