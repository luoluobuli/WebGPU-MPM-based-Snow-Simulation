// bukkitInsert.wgsl - Insert particles into sorted order by bukkit

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> bukkitParams: BukkitParams;

@group(1) @binding(0) var<storage, read_write> bukkitInsertCounters: array<atomic<u32>>;
@group(1) @binding(4) var<storage> bukkitIndexStart: array<u32>;

@group(2) @binding(0) var<storage, read_write> particles: array<ParticleData>;
@group(2) @binding(1) var<storage, read_write> sortedParticleIndices: array<u32>;

@compute @workgroup_size(256)
fn bukkitInsert(@builtin(global_invocation_id) gid: vec3u) {
    let particleIndex = gid.x;
    if (particleIndex >= bukkitParams.particleCount) { return; }
    
    let particle = particles[particleIndex];
    let bukkitId = positionToBukkitId(particle.pos);
    
    if (any(bukkitId < vec3i(0)) || 
        u32(bukkitId.x) >= bukkitParams.countX ||
        u32(bukkitId.y) >= bukkitParams.countY ||
        u32(bukkitId.z) >= bukkitParams.countZ) {
        return;
    }
    
    let bukkitIndex = bukkitIdToIndex(vec3u(bukkitId), bukkitParams.countX, bukkitParams.countY);
    let startIndex = bukkitIndexStart[bukkitIndex];
    let insertOffset = atomicAdd(&bukkitInsertCounters[bukkitIndex], 1u);
    
    sortedParticleIndices[startIndex + insertOffset] = particleIndex;
}
