@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid : SparseGridStorage;

@group(1) @binding(3) var<storage, read_write> grid_mass: array<i32>;
@group(1) @binding(4) var<storage, read_write> grid_momentum_x: array<i32>;
@group(1) @binding(5) var<storage, read_write> grid_momentum_y: array<i32>;
@group(1) @binding(6) var<storage, read_write> grid_momentum_z: array<i32>;




@compute
@workgroup_size(64) // run per block
fn doGridUpdate(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
) {
    let block_index = wid.y * 256 + wid.x;
    if block_index >= N_MAX_BLOCKS_IN_HASH_MAP { return; }
    
    let count = atomicLoad(&sparse_grid.n_allocated_blocks);
    if block_index >= count { return; }
    
    let mapped_block_index = sparse_grid.mapped_block_indexes[block_index];
    let block_number = sparse_grid.hash_map_entries[mapped_block_index].block_number;
    
    let cell_index_within_block = lid.x;
    let cell_index = block_index * 64u + cell_index_within_block;
    
    let cellMass = f32(grid_mass[cell_index]) / uniforms.fixedPointScale;
    if cellMass <= 0.0 { return; }

    if uniforms.use_pbmpm == 0 {
        let momX = f32(grid_momentum_x[cell_index]) / uniforms.fixedPointScale;
        let momY = f32(grid_momentum_y[cell_index]) / uniforms.fixedPointScale;
        let momZ = f32(grid_momentum_z[cell_index]) / uniforms.fixedPointScale;

        var v = vec3f(momX, momY, momZ) / cellMass;

        // ------------ Gravity -------------
        let gravity = vec3f(0.0, 0.0, -9.81);
        v += gravity * uniforms.simulationTimestep;
        


        let newMomentum = v * cellMass * uniforms.fixedPointScale;

        grid_momentum_x[cell_index] = i32(newMomentum.x);
        grid_momentum_y[cell_index] = i32(newMomentum.y);
        grid_momentum_z[cell_index] = i32(newMomentum.z);
    }
    
    else {
        let cell_momentum = vec3f(
            f32(grid_momentum_x[cell_index]) / uniforms.fixedPointScale,
            f32(grid_momentum_y[cell_index]) / uniforms.fixedPointScale,
            f32(grid_momentum_z[cell_index]) / uniforms.fixedPointScale,
        );

        let cell_mass = f32(grid_mass[cell_index]) / uniforms.fixedPointScale;

        var cell_velocity = cell_momentum / cell_mass;

        let gravitational_acceleration = vec3f(0, 0, -9.81) / 3;
        cell_velocity += gravitational_acceleration * uniforms.simulationTimestep;

        // Interaction
        if (uniforms.isInteracting != 0u) {
            let cell_coord = vec3f(
                f32(block_number.x * 4 + i32(lid.x % 4u)) - 0.5, // Approx? Wait.
                // Reconstruct cell coord from block index and lid
                // lid is 0..63.
                // block_number is vec3i.
                // block is 4x4x4.
                0.0, 0.0 // placeholder
            );
             // We need real cell coords.
        }
        // Wait, I need to reconstruct cell coords.
        // lid.x is cell_index_within_block (0-63).
        // sparseGridPrelude has helpers?
        // No, I have block_number.
        
        let cell_offset = vec3f(
            f32(lid.x % 4u),
            f32((lid.x / 4u) % 4u),
            f32(lid.x / 16u)
        );
        let grid_node_pos = vec3f(block_number * 4) + cell_offset;

        if (uniforms.isInteracting != 0u) {
            let dist = distance(grid_node_pos, uniforms.interactionPos);
            if (dist < uniforms.interactionRadius) {
                let offset = grid_node_pos - uniforms.interactionPos;
                var dir = vec3f(0.0, 0.0, 1.0);
                if (length(offset) > 0.001) {
                    dir = normalize(offset);
                }
                
                let falloff = 1.0 - (dist / uniforms.interactionRadius);
                let accel = dir * uniforms.interactionStrength * falloff / 3.0; // Divide by 3 for iterations
                
                cell_velocity += accel * uniforms.simulationTimestep;
            }
        }
        


        let new_momentum = cell_velocity * cell_mass * uniforms.fixedPointScale;

        grid_momentum_x[cell_index] = i32(new_momentum.x);
        grid_momentum_y[cell_index] = i32(new_momentum.y);
        grid_momentum_z[cell_index] = i32(new_momentum.z);
    }
}