// bukkitCount.wgsl - Count particles per bukkit

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> bukkitParams: BukkitParams;

@group(1) @binding(0) var<storage, read_write> bukkitCounts: array<atomic<u32>>;

@group(2) @binding(0) var<storage, read_write> particles: array<ParticleData>;

@compute @workgroup_size(256)
fn bukkitCount(@builtin(global_invocation_id) gid: vec3u) {
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
    atomicAdd(&bukkitCounts[bukkitIndex], 1u);
}
