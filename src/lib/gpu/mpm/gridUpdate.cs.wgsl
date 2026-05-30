@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid: SparseGridStorage;

@group(1) @binding(3) var<storage, read_write> grid_mass: array<i32>;
@group(1) @binding(4) var<storage, read_write> grid_momentum_x: array<i32>;
@group(1) @binding(5) var<storage, read_write> grid_momentum_y: array<i32>;
@group(1) @binding(6) var<storage, read_write> grid_momentum_z: array<i32>;

fn cellCenterGridCoord(block_number: vec3i, cell_index_within_block: u32) -> vec3f {
    let cell_offset = vec3f(
        f32(cell_index_within_block % BLOCK_SIZE),
        f32((cell_index_within_block / BLOCK_SIZE) % BLOCK_SIZE),
        f32(cell_index_within_block / (BLOCK_SIZE * BLOCK_SIZE)),
    );

    return vec3f(block_number * i32(BLOCK_SIZE)) + cell_offset + vec3f(0.5);
}

fn applyExternalGridForces(
    velocity: vec3f,
    block_number: vec3i,
    cell_index_within_block: u32,
) -> vec3f {
    var updated_velocity = velocity + vec3f(0.0, 0.0, -9.81) * uniforms.simulationTimestep;

    if uniforms.isInteracting != 0u && uniforms.interactionRadius > 0.0 {
        let grid_pos = cellCenterGridCoord(block_number, cell_index_within_block);
        let offset = grid_pos - uniforms.interactionPos;
        let dist = length(offset);

        if dist < uniforms.interactionRadius {
            var dir = vec3f(0.0, 0.0, 1.0);
            if dist > 0.001 {
                dir = offset / dist;
            }

            let falloff_linear = dist / uniforms.interactionRadius;
            let falloff = 1.0 - falloff_linear * falloff_linear;
            let signed_strength = select(1.0, -1.0, uniforms.interactionMode == 1u);
            updated_velocity += signed_strength * dir * uniforms.interactionStrength * falloff * uniforms.simulationTimestep;
        }
    }

    return updated_velocity;
}

fn applyDomainBoundaryVelocity(
    velocity: vec3f,
    block_number: vec3i,
    cell_index_within_block: u32,
) -> vec3f {
    let cell = block_number * i32(BLOCK_SIZE) + vec3i(
        i32(cell_index_within_block % BLOCK_SIZE),
        i32((cell_index_within_block / BLOCK_SIZE) % BLOCK_SIZE),
        i32(cell_index_within_block / (BLOCK_SIZE * BLOCK_SIZE)),
    );

    var bounded_velocity = velocity;
    let boundary_width = 2i;
    if cell.x < boundary_width && bounded_velocity.x < 0.0 { bounded_velocity.x = 0.0; }
    if cell.x > i32(uniforms.gridResolution.x) - boundary_width - 1i && bounded_velocity.x > 0.0 { bounded_velocity.x = 0.0; }
    if cell.y < boundary_width && bounded_velocity.y < 0.0 { bounded_velocity.y = 0.0; }
    if cell.y > i32(uniforms.gridResolution.y) - boundary_width - 1i && bounded_velocity.y > 0.0 { bounded_velocity.y = 0.0; }
    if cell.z < boundary_width && bounded_velocity.z < 0.0 { bounded_velocity.z = 0.0; }
    if cell.z > i32(uniforms.gridResolution.z) - boundary_width - 1i && bounded_velocity.z > 0.0 { bounded_velocity.z = 0.0; }

    return bounded_velocity;
}

fn limitVelocityToCfl(velocity: vec3f) -> vec3f {
    return clampVec3Length(velocity, maxStableParticleSpeed());
}

@compute
@workgroup_size(64)
fn doGridUpdate(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
) {
    let block_index = wid.y * 256u + wid.x;
    if block_index >= N_MAX_BLOCKS_IN_HASH_MAP { return; }
    
    let count = atomicLoad(&sparse_grid.n_allocated_blocks);
    if block_index >= count { return; }
    
    let mapped_block_index = sparse_grid.mapped_block_indexes[block_index];
    let block_number = sparse_grid.hash_map_entries[mapped_block_index].block_number;
    
    let cell_index_within_block = lid.x;
    let cell_index = block_index * BLOCK_SIZE_CUBED + cell_index_within_block;
    
    let cell_mass = f32(grid_mass[cell_index]) / uniforms.fixedPointScale;
    if cell_mass <= 0.0 { return; }

    let cell_momentum = vec3f(
        f32(grid_momentum_x[cell_index]) / uniforms.fixedPointScale,
        f32(grid_momentum_y[cell_index]) / uniforms.fixedPointScale,
        f32(grid_momentum_z[cell_index]) / uniforms.fixedPointScale,
    );

    var cell_velocity = applyExternalGridForces(
        cell_momentum / cell_mass,
        block_number,
        cell_index_within_block,
    );
    cell_velocity = applyDomainBoundaryVelocity(cell_velocity, block_number, cell_index_within_block);
    cell_velocity = limitVelocityToCfl(cell_velocity);

    let new_momentum = cell_velocity * cell_mass;

    grid_momentum_x[cell_index] = toFixedPointI32(new_momentum.x);
    grid_momentum_y[cell_index] = toFixedPointI32(new_momentum.y);
    grid_momentum_z[cell_index] = toFixedPointI32(new_momentum.z);
}
