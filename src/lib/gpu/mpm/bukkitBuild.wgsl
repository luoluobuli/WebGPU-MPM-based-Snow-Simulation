@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid: SparseGridStorage;

@group(2) @binding(0) var<storage, read> particle_data: array<ParticleData>;

struct BukkitThreadData {
    range_start: u32,
    range_count: u32,
    origin_cell_x: u32,
    origin_cell_y: u32,
    origin_cell_z: u32,
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
override FUSED_PARTICLE_WORKGROUP_SIZE: u32 = 64u;

const FUSED_BUKKIT_SIZE = 2u;
const FUSED_BUKKIT_SUBDIVISIONS_PER_BLOCK = BLOCK_SIZE / FUSED_BUKKIT_SIZE;
const FUSED_BUKKIT_SUBBUKKITS_PER_BLOCK = FUSED_BUKKIT_SUBDIVISIONS_PER_BLOCK
    * FUSED_BUKKIT_SUBDIVISIONS_PER_BLOCK
    * FUSED_BUKKIT_SUBDIVISIONS_PER_BLOCK;
const BUKKIT_DISPATCH_WIDTH = 256u;

fn sourceActiveBlockCount() -> u32 {
    return min(atomicLoad(&sparse_grid.n_allocated_blocks), N_MAX_ACTIVE_BLOCKS);
}

fn startCellNumberFromParticlePosition(pos: vec3f) -> vec3i {
    let grid_coord = calculateGridCoordinate(pos);
    return vec3i(floor(grid_coord));
}

fn sourceGridBlockIndex(block_number: vec3i) -> u32 {
    if !bukkitCanContainBlock(block_number) {
        return GRID_BLOCK_INDEX_EMPTY;
    }

    let bukkit_index = calculateBukkitIndex(block_number);
    if atomicLoad(&sparse_grid.bukkit_generations[bukkit_index]) != sparse_grid.current_generation {
        return GRID_BLOCK_INDEX_EMPTY;
    }

    let block_index = atomicLoad(&sparse_grid.block_index_bukkits[bukkit_index]);
    if block_index >= N_MAX_ACTIVE_BLOCKS {
        return GRID_BLOCK_INDEX_EMPTY;
    }

    return block_index;
}

fn subbukkitIndexFromStartCell(start_cell_number: vec3i) -> u32 {
    let local_cell = vec3u(start_cell_number & vec3i(i32(BLOCK_MASK)));
    let subbukkit = local_cell / FUSED_BUKKIT_SIZE;
    return subbukkit.x
        + FUSED_BUKKIT_SUBDIVISIONS_PER_BLOCK * (
            subbukkit.y + FUSED_BUKKIT_SUBDIVISIONS_PER_BLOCK * subbukkit.z
        );
}

fn subbukkitOffsetFromIndex(subbukkit_index: u32) -> vec3u {
    return vec3u(
        subbukkit_index % FUSED_BUKKIT_SUBDIVISIONS_PER_BLOCK,
        (subbukkit_index / FUSED_BUKKIT_SUBDIVISIONS_PER_BLOCK) % FUSED_BUKKIT_SUBDIVISIONS_PER_BLOCK,
        subbukkit_index / (
            FUSED_BUKKIT_SUBDIVISIONS_PER_BLOCK * FUSED_BUKKIT_SUBDIVISIONS_PER_BLOCK
        ),
    );
}

fn bukkitIndexFromActiveBlockAndSubbukkit(active_block_index: u32, subbukkit_index: u32) -> u32 {
    return active_block_index * FUSED_BUKKIT_SUBBUKKITS_PER_BLOCK + subbukkit_index;
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

    if thread_index >= sourceActiveBlockCount() { return; }

    let block_number = sparse_grid.mapped_block_numbers[thread_index];
    if !bukkitCanContainBlock(block_number) { return; }

    let bukkit_base_index = thread_index * FUSED_BUKKIT_SUBBUKKITS_PER_BLOCK;
    for (var subbukkit_index = 0u; subbukkit_index < FUSED_BUKKIT_SUBBUKKITS_PER_BLOCK; subbukkit_index = subbukkit_index + 1u) {
        let bukkit_index = bukkit_base_index + subbukkit_index;
        atomicStore(&bukkit_particle_counts[bukkit_index], 0u);
        atomicStore(&bukkit_insert_counters[bukkit_index], 0u);
    }
}

@compute
@workgroup_size(PARTICLE_WORKGROUP_SIZE)
fn countParticlesPerBukkit(@builtin(global_invocation_id) gid: vec3u) {
    let particle_index = gid.x;
    if particle_index >= N_PARTICLES { return; }

    let particle_pos = particle_data[particle_index].pos;
    if !particlePositionCanTouchGrid(particle_pos) { return; }

    let start_cell_number = startCellNumberFromParticlePosition(particle_pos);
    let block_number = calculateBlockNumberContainingCell(start_cell_number);
    let active_block_index = sourceGridBlockIndex(block_number);
    if active_block_index == GRID_BLOCK_INDEX_EMPTY { return; }

    let subbukkit_index = subbukkitIndexFromStartCell(start_cell_number);
    atomicAdd(
        &bukkit_particle_counts[
            bukkitIndexFromActiveBlockAndSubbukkit(active_block_index, subbukkit_index)
        ],
        1u,
    );
}

@compute
@workgroup_size(256)
fn allocateBukkitThreadData(@builtin(global_invocation_id) gid: vec3u) {
    let active_block_index = gid.x;
    if active_block_index >= sourceActiveBlockCount() { return; }

    let block_number = sparse_grid.mapped_block_numbers[active_block_index];
    if !bukkitCanContainBlock(block_number) { return; }

    let block_cell_base = vec3u(block_number) * BLOCK_SIZE;
    let bukkit_base_index = active_block_index * FUSED_BUKKIT_SUBBUKKITS_PER_BLOCK;

    for (var subbukkit_index = 0u; subbukkit_index < FUSED_BUKKIT_SUBBUKKITS_PER_BLOCK; subbukkit_index = subbukkit_index + 1u) {
        let bukkit_index = bukkit_base_index + subbukkit_index;
        let bukkit_count = atomicLoad(&bukkit_particle_counts[bukkit_index]);
        if bukkit_count == 0u { continue; }

        let dispatch_count = (bukkit_count + FUSED_PARTICLE_WORKGROUP_SIZE - 1u) / FUSED_PARTICLE_WORKGROUP_SIZE;
        let dispatch_start = atomicAdd(&bukkit_dispatch_args.count, dispatch_count);
        let particle_start = atomicAdd(&bukkit_particle_allocator.count, bukkit_count);
        let origin_cell = block_cell_base
            + subbukkitOffsetFromIndex(subbukkit_index) * FUSED_BUKKIT_SIZE;

        bukkit_index_start[bukkit_index] = particle_start;

        for (var i = 0u; i < dispatch_count; i = i + 1u) {
            let range_start = particle_start + i * FUSED_PARTICLE_WORKGROUP_SIZE;
            let particles_remaining = bukkit_count - i * FUSED_PARTICLE_WORKGROUP_SIZE;
            let range_count = min(FUSED_PARTICLE_WORKGROUP_SIZE, particles_remaining);
            bukkit_thread_data[dispatch_start + i] = BukkitThreadData(
                range_start,
                range_count,
                origin_cell.x,
                origin_cell.y,
                origin_cell.z,
                0u,
                0u,
                0u,
            );
        }
    }
}

@compute
@workgroup_size(PARTICLE_WORKGROUP_SIZE)
fn insertParticlesIntoBukkit(@builtin(global_invocation_id) gid: vec3u) {
    let particle_index = gid.x;
    if particle_index >= N_PARTICLES { return; }

    let particle_pos = particle_data[particle_index].pos;
    if !particlePositionCanTouchGrid(particle_pos) { return; }

    let start_cell_number = startCellNumberFromParticlePosition(particle_pos);
    let block_number = calculateBlockNumberContainingCell(start_cell_number);
    let active_block_index = sourceGridBlockIndex(block_number);
    if active_block_index == GRID_BLOCK_INDEX_EMPTY { return; }

    let subbukkit_index = subbukkitIndexFromStartCell(start_cell_number);
    let bukkit_index = bukkitIndexFromActiveBlockAndSubbukkit(active_block_index, subbukkit_index);
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
