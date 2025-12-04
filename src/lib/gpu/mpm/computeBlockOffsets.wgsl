@group(1) @binding(1) var<storage, read_write> n_allocated_blocks: atomic<u32>;
@group(1) @binding(7) var<storage, read_write> block_particle_counts: array<u32>;
@group(1) @binding(8) var<storage, read_write> block_particle_offsets: array<u32>;

@compute
@workgroup_size(1)
fn computeBlockOffsets() {
    let count = atomicLoad(&n_allocated_blocks);
    var offset = 0u;
    for (var i = 0u; i < count; i++) {
        let c = block_particle_counts[i];
        block_particle_offsets[i] = offset;
        offset += c;
    }
}
