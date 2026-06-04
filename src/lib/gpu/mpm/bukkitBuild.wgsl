@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid: SparseGridStorage;

@group(2) @binding(0) var<storage, read> particle_data: array<ParticleData>;
@group(2) @binding(3) var<storage, read> active_block_dispatch_args: array<u32>;

struct BukkitThreadData {
    range_start: u32,
    range_count: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct BukkitDispatchArgs {
    x: atomic<u32>,
    y: atomic<u32>,
    z: atomic<u32>,
    count: atomic<u32>,
}

struct BukkitParticleAllocator {
    count: atomic<u32>,
}

struct BukkitThreadGroupCount {
    count: atomic<u32>,
}

@group(3) @binding(2) var<storage, read_write> bukkit_particle_counts: array<atomic<u32>>;
@group(3) @binding(3) var<storage, read_write> bukkit_insert_counters: array<atomic<u32>>;
@group(3) @binding(4) var<storage, read_write> bukkit_index_start: array<u32>;
@group(3) @binding(5) var<storage, read_write> bukkit_thread_data: array<BukkitThreadData>;
@group(3) @binding(6) var<storage, read_write> bukkit_particle_data: array<u32>;
@group(3) @binding(7) var<storage, read_write> bukkit_dispatch_args: BukkitDispatchArgs;
@group(3) @binding(8) var<storage, read_write> bukkit_particle_allocator: BukkitParticleAllocator;
@group(3) @binding(9) var<storage, read_write> bukkit_thread_group_count: BukkitThreadGroupCount;

override N_PARTICLES: u32 = 0u;
override PARTICLE_WORKGROUP_SIZE: u32 = 256u;

const BUKKIT_DISPATCH_WIDTH = 256u;

fn blockNumberFromParticlePosition(pos: vec3f) -> vec3i {
    let grid_coord = calculateGridCoordinate(pos);
    let start_cell_number = vec3i(floor(grid_coord));
    return calculateBlockNumberContainingCell(start_cell_number);
}

fn sourceGridContainsBlock(block_number: vec3i) -> bool {
    if !bukkitCanContainBlock(block_number) {
        return false;
    }

    let bukkit_index = calculateBukkitIndex(block_number);
    return atomicLoad(&sparse_grid.bukkit_generations[bukkit_index])
        == sparse_grid.current_generation;
}

@compute
@workgroup_size(256)
fn resetBukkitBuildBuffers(@builtin(global_invocation_id) gid: vec3u) {
    let thread_index = gid.x;

    if thread_index == 0u {
        atomicStore(&bukkit_dispatch_args.x, 0u);
        atomicStore(&bukkit_dispatch_args.y, 1u);
        atomicStore(&bukkit_dispatch_args.z, 1u);
        atomicStore(&bukkit_dispatch_args.count, 0u);
        atomicStore(&bukkit_particle_allocator.count, 0u);
        atomicStore(&bukkit_thread_group_count.count, 0u);
    }

    if thread_index >= active_block_dispatch_args[3] { return; }

    let block_number = sparse_grid.mapped_block_numbers[thread_index];
    if !bukkitCanContainBlock(block_number) { return; }

    let bukkit_index = calculateBukkitIndex(block_number);

    atomicStore(&bukkit_particle_counts[bukkit_index], 0u);
    atomicStore(&bukkit_insert_counters[bukkit_index], 0u);
}

@compute
@workgroup_size(PARTICLE_WORKGROUP_SIZE)
fn countParticlesPerBukkit(@builtin(global_invocation_id) gid: vec3u) {
    let particle_index = gid.x;
    if particle_index >= N_PARTICLES { return; }

    let particle_pos = particle_data[particle_index].pos;
    if !particlePositionCanTouchGrid(particle_pos) { return; }

    let block_number = blockNumberFromParticlePosition(particle_pos);
    if !sourceGridContainsBlock(block_number) { return; }

    atomicAdd(&bukkit_particle_counts[calculateBukkitIndex(block_number)], 1u);
}

fn blockNumberFromBukkitIndex(bukkit_index: u32) -> vec3u {
    let block_x = bukkit_index % BUKKIT_DOMAIN_BLOCKS_X;
    let block_y = (bukkit_index / BUKKIT_DOMAIN_BLOCKS_X) % BUKKIT_DOMAIN_BLOCKS_Y;
    let block_z = bukkit_index / (BUKKIT_DOMAIN_BLOCKS_X * BUKKIT_DOMAIN_BLOCKS_Y);
    return vec3u(block_x, block_y, block_z);
}

@compute
@workgroup_size(256)
fn allocateBukkitThreadData(@builtin(global_invocation_id) gid: vec3u) {
    let active_block_index = gid.x;
    if active_block_index >= active_block_dispatch_args[3] { return; }

    let block_number_i = sparse_grid.mapped_block_numbers[active_block_index];
    if !bukkitCanContainBlock(block_number_i) { return; }

    let bukkit_index = calculateBukkitIndex(block_number_i);

    let bukkit_count = atomicLoad(&bukkit_particle_counts[bukkit_index]);
    if bukkit_count == 0u { return; }

    let dispatch_count = (bukkit_count + PARTICLE_WORKGROUP_SIZE - 1u) / PARTICLE_WORKGROUP_SIZE;
    let dispatch_start = atomicAdd(&bukkit_dispatch_args.count, dispatch_count);
    let particle_start = atomicAdd(&bukkit_particle_allocator.count, bukkit_count);
    let block_number = vec3u(block_number_i);

    bukkit_index_start[bukkit_index] = particle_start;

    for (var i = 0u; i < dispatch_count; i = i + 1u) {
        let range_start = particle_start + i * PARTICLE_WORKGROUP_SIZE;
        let particles_remaining = bukkit_count - i * PARTICLE_WORKGROUP_SIZE;
        let range_count = min(PARTICLE_WORKGROUP_SIZE, particles_remaining);
        bukkit_thread_data[dispatch_start + i] = BukkitThreadData(
            range_start,
            range_count,
            block_number.x,
            block_number.y,
            block_number.z,
            0u,
            0u,
            0u,
        );
    }
}

@compute
@workgroup_size(PARTICLE_WORKGROUP_SIZE)
fn insertParticlesIntoBukkit(@builtin(global_invocation_id) gid: vec3u) {
    let particle_index = gid.x;
    if particle_index >= N_PARTICLES { return; }

    let particle_pos = particle_data[particle_index].pos;
    if !particlePositionCanTouchGrid(particle_pos) { return; }

    let block_number = blockNumberFromParticlePosition(particle_pos);
    if !sourceGridContainsBlock(block_number) { return; }

    let bukkit_index = calculateBukkitIndex(block_number);
    let index_start = bukkit_index_start[bukkit_index];
    let particle_slot = atomicAdd(&bukkit_insert_counters[bukkit_index], 1u);
    bukkit_particle_data[index_start + particle_slot] = particle_index;
}

@compute
@workgroup_size(1)
fn finalizeBukkitDispatch() {
    let dispatch_count = atomicLoad(&bukkit_dispatch_args.count);
    atomicStore(&bukkit_thread_group_count.count, dispatch_count);
    if dispatch_count == 0u {
        atomicStore(&bukkit_dispatch_args.x, 1u);
        atomicStore(&bukkit_dispatch_args.y, 1u);
        atomicStore(&bukkit_dispatch_args.z, 1u);
        return;
    }

    atomicStore(&bukkit_dispatch_args.x, min(dispatch_count, BUKKIT_DISPATCH_WIDTH));
    atomicStore(
        &bukkit_dispatch_args.y,
        (dispatch_count + BUKKIT_DISPATCH_WIDTH - 1u) / BUKKIT_DISPATCH_WIDTH,
    );
    atomicStore(&bukkit_dispatch_args.z, 1u);
}
