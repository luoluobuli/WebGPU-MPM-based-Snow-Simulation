
const BLOCK_SIZE = 4;
const BUKKIT_DOMAIN_GRID_RESOLUTION = 384;
export const BUKKIT_DOMAIN_BLOCKS_PER_AXIS = Math.ceil(BUKKIT_DOMAIN_GRID_RESOLUTION / BLOCK_SIZE);
export const BUKKIT_DOMAIN_BLOCK_COUNT = BUKKIT_DOMAIN_BLOCKS_PER_AXIS ** 3;
export const N_MAX_ACTIVE_BLOCKS = 100_000;
const GRID_CELLS_PER_BLOCK = 64;
const SPARSE_GRID_HEADER_BYTES = 16;
const UINT32_BYTES = 4;
const VEC3I_STORAGE_STRIDE_BYTES = 16;
const BUKKIT_THREAD_DATA_BYTES = 32;

function alignTo(value: number, alignment: number) {
    return Math.ceil(value / alignment) * alignment;
}

export class GpuMpmBufferManager {
    readonly particleDataBuffer: GPUBuffer;
    readonly particleFlagsBuffer: GPUBuffer;
    readonly sparseGridBuffer: GPUBuffer;
    readonly nextSparseGridBuffer: GPUBuffer;
    readonly gridAccumulatorBuffer: GPUBuffer;
    readonly gridVelocityBuffer: GPUBuffer;
    readonly activeBlockDispatchBuffer: GPUBuffer;
    readonly nextActiveBlockDispatchBuffer: GPUBuffer;
    readonly bukkitParticleCountsBuffer: GPUBuffer;
    readonly bukkitInsertCountersBuffer: GPUBuffer;
    readonly bukkitIndexStartBuffer: GPUBuffer;
    readonly bukkitThreadDataBuffer: GPUBuffer;
    readonly bukkitParticleDataBuffer: GPUBuffer;
    readonly bukkitDispatchBuffer: GPUBuffer;
    readonly bukkitParticleAllocatorBuffer: GPUBuffer;
    readonly bukkitThreadGroupCountBuffer: GPUBuffer;

    readonly nParticles: number;
    readonly nMaxActiveBlocks = N_MAX_ACTIVE_BLOCKS;
    readonly bukkitDomainBlockCount = BUKKIT_DOMAIN_BLOCK_COUNT;

    constructor({
        device,
        nParticles,
    }: {
        device: GPUDevice,
        nParticles: number,
    }) {
        const particleDataBuffer = device.createBuffer({
            label: "MPM particle data buffer",
            size: nParticles * 192,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
        });

        const particleFlagsBuffer = device.createBuffer({
            label: "MPM particle flags buffer",
            size: nParticles * UINT32_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // SparseGridStorage layout:
        // - n_allocated_blocks + current_generation: u32 * 2 + 8 bytes explicit padding = 16 bytes
        // - block_index_bukkits: array<atomic<u32>, BUKKIT_DOMAIN_BLOCK_COUNT>
        // - bukkit_generations: array<atomic<u32>, BUKKIT_DOMAIN_BLOCK_COUNT>
        // - mapped_block_numbers: array<vec3i, N_MAX_ACTIVE_BLOCKS> with 16-byte storage stride
        const mappedBlockNumbersOffset = alignTo(
            SPARSE_GRID_HEADER_BYTES + this.bukkitDomainBlockCount * UINT32_BYTES * 2,
            VEC3I_STORAGE_STRIDE_BYTES,
        );
        const sparseGridBufferSize = mappedBlockNumbersOffset
            + this.nMaxActiveBlocks * VEC3I_STORAGE_STRIDE_BYTES;
        const createSparseGridBuffer = (label: string) => device.createBuffer({
            label,
            size: sparseGridBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const sparseGridBuffer = createSparseGridBuffer("MPM sparse grid storage buffer");
        const nextSparseGridBuffer = createSparseGridBuffer("MPM fused next sparse grid storage buffer");

        const gridVectorStorageSize = this.nMaxActiveBlocks * GRID_CELLS_PER_BLOCK * VEC3I_STORAGE_STRIDE_BYTES;
        const gridAccumulatorBuffer = device.createBuffer({
            label: "MPM packed grid accumulator buffer",
            size: gridVectorStorageSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const gridVelocityBuffer = device.createBuffer({
            label: "MPM packed grid velocity buffer",
            size: gridVectorStorageSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // 0..2: active block dispatch, 3: active block count.
        const activeBlockDispatchBuffer = device.createBuffer({
            label: "MPM active block indirect dispatch buffer",
            size: 4 * UINT32_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        });
        const nextActiveBlockDispatchBuffer = device.createBuffer({
            label: "MPM fused next active block indirect dispatch buffer",
            size: 4 * UINT32_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        });

        const bukkitTableSize = this.bukkitDomainBlockCount * UINT32_BYTES;
        const bukkitParticleCountsBuffer = device.createBuffer({
            label: "MPM fused bukkit particle counts buffer",
            size: bukkitTableSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const bukkitInsertCountersBuffer = device.createBuffer({
            label: "MPM fused bukkit insert counters buffer",
            size: bukkitTableSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const bukkitIndexStartBuffer = device.createBuffer({
            label: "MPM fused bukkit index starts buffer",
            size: bukkitTableSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const bukkitThreadDataBuffer = device.createBuffer({
            label: "MPM fused bukkit thread data buffer",
            size: Math.max(1, nParticles) * BUKKIT_THREAD_DATA_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const bukkitParticleDataBuffer = device.createBuffer({
            label: "MPM fused bukkit particle index buffer",
            size: Math.max(1, nParticles) * UINT32_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const bukkitDispatchBuffer = device.createBuffer({
            label: "MPM fused bukkit indirect dispatch buffer",
            size: 4 * UINT32_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        });
        const bukkitParticleAllocatorBuffer = device.createBuffer({
            label: "MPM fused bukkit particle allocator buffer",
            size: UINT32_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const bukkitThreadGroupCountBuffer = device.createBuffer({
            label: "MPM fused bukkit thread group count buffer",
            size: UINT32_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.particleDataBuffer = particleDataBuffer;
        this.particleFlagsBuffer = particleFlagsBuffer;
        this.sparseGridBuffer = sparseGridBuffer;
        this.nextSparseGridBuffer = nextSparseGridBuffer;
        this.gridAccumulatorBuffer = gridAccumulatorBuffer;
        this.gridVelocityBuffer = gridVelocityBuffer;
        this.activeBlockDispatchBuffer = activeBlockDispatchBuffer;
        this.nextActiveBlockDispatchBuffer = nextActiveBlockDispatchBuffer;
        this.bukkitParticleCountsBuffer = bukkitParticleCountsBuffer;
        this.bukkitInsertCountersBuffer = bukkitInsertCountersBuffer;
        this.bukkitIndexStartBuffer = bukkitIndexStartBuffer;
        this.bukkitThreadDataBuffer = bukkitThreadDataBuffer;
        this.bukkitParticleDataBuffer = bukkitParticleDataBuffer;
        this.bukkitDispatchBuffer = bukkitDispatchBuffer;
        this.bukkitParticleAllocatorBuffer = bukkitParticleAllocatorBuffer;
        this.bukkitThreadGroupCountBuffer = bukkitThreadGroupCountBuffer;

        this.nParticles = nParticles;
    }
}
