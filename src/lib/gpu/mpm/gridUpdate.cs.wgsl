@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid: SparseGridStorage;

@group(1) @binding(3) var<storage, read_write> grid_accumulator: array<vec4i>;

struct GridVelocity {
    velocity: vec3f,
}

@group(1) @binding(7) var<storage, read_write> grid_velocity: array<GridVelocity>;

@group(2) @binding(0) var<storage, read> active_block_dispatch_args: array<u32>;

override ENABLE_INTERACTION: u32 = 1u;
override GRID_BOUNDARY_MAX_X: i32 = 381i;
override GRID_BOUNDARY_MAX_Y: i32 = 381i;
override GRID_BOUNDARY_MAX_Z: i32 = 381i;
override GRID_BOUNDARY_HIGH_BLOCK_X: i32 = 95i;
override GRID_BOUNDARY_HIGH_BLOCK_Y: i32 = 95i;
override GRID_BOUNDARY_HIGH_BLOCK_Z: i32 = 95i;

const GRID_BOUNDARY_WIDTH = 2i;
const ACTIVE_BLOCK_DISPATCH_WIDTH = 256u;

var<workgroup> workgroup_block_cell_base: vec3i;
var<workgroup> workgroup_is_interior_block: bool;
var<workgroup> workgroup_has_active_block: bool;
var<workgroup> workgroup_boundary_low_end: vec3u;
var<workgroup> workgroup_boundary_high_start: vec3u;

fn cellOffsetWithinBlock(cell_index_within_block: u32) -> vec3u {
    return vec3u(
        cell_index_within_block & BLOCK_MASK,
        (cell_index_within_block >> LOG_BLOCK_SIZE) & BLOCK_MASK,
        cell_index_within_block >> (LOG_BLOCK_SIZE * 2u),
    );
}

fn cellCenterGridCoord(block_cell_base: vec3i, cell_offset: vec3u) -> vec3f {
    return vec3f(block_cell_base) + vec3f(cell_offset) + vec3f(0.5);
}

fn boundaryLowEnd(block_axis: i32) -> u32 {
    return u32(clamp(
        GRID_BOUNDARY_WIDTH - block_axis * i32(BLOCK_SIZE),
        0i,
        i32(BLOCK_SIZE),
    ));
}

fn boundaryHighStart(block_axis: i32, boundary_max: i32) -> u32 {
    return u32(clamp(
        boundary_max - block_axis * i32(BLOCK_SIZE) + 1i,
        0i,
        i32(BLOCK_SIZE),
    ));
}

fn applyInteractionVelocity(
    velocity: vec3f,
    block_cell_base: vec3i,
    cell_offset: vec3u,
) -> vec3f {
    let grid_pos = cellCenterGridCoord(block_cell_base, cell_offset);
    let offset = grid_pos - uniforms.interactionPos;
    let dist_squared = dot(offset, offset);
    let radius_squared = uniforms.interactionRadiusSquared;

    if dist_squared < radius_squared {
        var dir = vec3f(0.0, 0.0, 1.0);
        if dist_squared > 1e-6 {
            dir = offset * inverseSqrt(dist_squared);
        }

        let falloff = 1.0 - dist_squared / radius_squared;
        let signed_strength = select(1.0, -1.0, uniforms.interactionMode == 1u);
        return velocity + signed_strength * dir * uniforms.interactionStrengthDelta * falloff;
    }

    return velocity;
}

fn applyDomainBoundaryVelocity(
    velocity: vec3f,
    cell_offset: vec3u,
) -> vec3f {
    var bounded_velocity = velocity;
    if cell_offset.x < workgroup_boundary_low_end.x && bounded_velocity.x < 0.0 { bounded_velocity.x = 0.0; }
    if cell_offset.x >= workgroup_boundary_high_start.x && bounded_velocity.x > 0.0 { bounded_velocity.x = 0.0; }
    if cell_offset.y < workgroup_boundary_low_end.y && bounded_velocity.y < 0.0 { bounded_velocity.y = 0.0; }
    if cell_offset.y >= workgroup_boundary_high_start.y && bounded_velocity.y > 0.0 { bounded_velocity.y = 0.0; }
    if cell_offset.z < workgroup_boundary_low_end.z && bounded_velocity.z < 0.0 { bounded_velocity.z = 0.0; }
    if cell_offset.z >= workgroup_boundary_high_start.z && bounded_velocity.z > 0.0 { bounded_velocity.z = 0.0; }

    return bounded_velocity;
}

fn limitVelocityToCfl(velocity: vec3f) -> vec3f {
    let len_squared = dot(velocity, velocity);
    let max_len_squared = uniforms.maxStableParticleSpeedSquared;
    if len_squared > max_len_squared {
        return velocity * (uniforms.maxStableParticleSpeed * inverseSqrt(len_squared));
    }

    return velocity;
}

@compute
@workgroup_size(64)
fn doGridUpdate(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
) {
    let block_index = wid.y * ACTIVE_BLOCK_DISPATCH_WIDTH + wid.x;
    if lid.x == 0u {
        let has_active_block = block_index < active_block_dispatch_args[3];
        workgroup_has_active_block = has_active_block;

        if has_active_block {
            let block_number = sparse_grid.mapped_block_numbers[block_index];
            workgroup_block_cell_base = block_number * i32(BLOCK_SIZE);
            workgroup_is_interior_block = block_number.x > 0i && block_number.x < GRID_BOUNDARY_HIGH_BLOCK_X
                && block_number.y > 0i && block_number.y < GRID_BOUNDARY_HIGH_BLOCK_Y
                && block_number.z > 0i && block_number.z < GRID_BOUNDARY_HIGH_BLOCK_Z;
            if !workgroup_is_interior_block {
                workgroup_boundary_low_end = vec3u(
                    boundaryLowEnd(block_number.x),
                    boundaryLowEnd(block_number.y),
                    boundaryLowEnd(block_number.z),
                );
                workgroup_boundary_high_start = vec3u(
                    boundaryHighStart(block_number.x, GRID_BOUNDARY_MAX_X),
                    boundaryHighStart(block_number.y, GRID_BOUNDARY_MAX_Y),
                    boundaryHighStart(block_number.z, GRID_BOUNDARY_MAX_Z),
                );
            }
        }
    }
    workgroupBarrier();

    if !workgroup_has_active_block { return; }

    let block_cell_base = workgroup_block_cell_base;

    let cell_index_within_block = lid.x;
    let cell_index = (block_index << LOG_BLOCK_SIZE_CUBED) + cell_index_within_block;

    let cell_accumulator = grid_accumulator[cell_index];
    grid_accumulator[cell_index] = vec4i(0);

    let cell_mass_fixed_i32 = cell_accumulator.x;
    if cell_mass_fixed_i32 <= 0i {
        grid_velocity[cell_index].velocity = vec3f(0.0);
        return;
    }
    let cell_mass_fixed = f32(cell_mass_fixed_i32);
    let inv_cell_mass_fixed = 1.0 / cell_mass_fixed;

    let cell_velocity_in = vec3f(cell_accumulator.yzw) * inv_cell_mass_fixed;

    var cell_velocity = cell_velocity_in + uniforms.gravityDeltaVelocity;
    if ENABLE_INTERACTION != 0u && uniforms.isInteracting != 0u && uniforms.interactionRadiusSquared > 0.0 {
        let cell_offset = cellOffsetWithinBlock(cell_index_within_block);
        cell_velocity = applyInteractionVelocity(cell_velocity, block_cell_base, cell_offset);
        if !workgroup_is_interior_block {
            cell_velocity = applyDomainBoundaryVelocity(cell_velocity, cell_offset);
        }
    } else if !workgroup_is_interior_block {
        let cell_offset = cellOffsetWithinBlock(cell_index_within_block);
        cell_velocity = applyDomainBoundaryVelocity(cell_velocity, cell_offset);
    }
    cell_velocity = limitVelocityToCfl(cell_velocity);

    grid_velocity[cell_index].velocity = cell_velocity;
}
