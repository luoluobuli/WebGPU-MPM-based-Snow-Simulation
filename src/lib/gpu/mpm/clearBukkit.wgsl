@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid : SparseGridStorage;

@compute
@workgroup_size(1)
fn clearBukkit(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let thread_index = gid.x;
    if thread_index != 0u { return; }

    let current_generation = sparse_grid.current_generation;
    var next_generation = current_generation + 1u;
    if next_generation == 0u || next_generation == BUKKIT_GENERATION_RESERVED {
        next_generation = 1u;
    }

    sparse_grid.current_generation = next_generation;
    atomicStore(&sparse_grid.n_allocated_blocks, 0u);
}
