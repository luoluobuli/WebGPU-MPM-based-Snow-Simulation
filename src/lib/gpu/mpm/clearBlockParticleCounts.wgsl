@group(1) @binding(7) var<storage, read_write> block_particle_counts: array<u32>;

@compute
@workgroup_size(256)
fn clearBlockParticleCounts(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let thread_index = gid.x;
    if thread_index < arrayLength(&block_particle_counts) {
        block_particle_counts[thread_index] = 0u;
    }
}
