@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@compute
@workgroup_size(1)
fn computeBlockOffsets() {
    let count = atomicLoad(&sparse_grid.n_allocated_blocks);
    var offset = 0u;
    for (var i = 0u; i < count; i++) {
        let c = atomicLoad(&sparse_grid.block_particle_counts[i]);
        atomicStore(&sparse_grid.block_particle_offsets[i], offset);
        offset += c;
    }
}
