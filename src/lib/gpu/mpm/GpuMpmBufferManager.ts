
const BLOCK_SIZE = 4;
const BUKKIT_DOMAIN_GRID_RESOLUTION = 384;
export const BUKKIT_DOMAIN_BLOCKS_PER_AXIS = Math.ceil(BUKKIT_DOMAIN_GRID_RESOLUTION / BLOCK_SIZE);
export const BUKKIT_DOMAIN_BLOCK_COUNT = BUKKIT_DOMAIN_BLOCKS_PER_AXIS ** 3;
const N_MAX_ACTIVE_BLOCKS = 100_000;
const GRID_CELLS_PER_BLOCK = 64;
const SPARSE_GRID_HEADER_BYTES = 16;
const UINT32_BYTES = 4;
const VEC3I_STORAGE_STRIDE_BYTES = 16;

function alignTo(value: number, alignment: number) {
    return Math.ceil(value / alignment) * alignment;
}

export class GpuMpmBufferManager {
    readonly particleDataBuffer: GPUBuffer;
    readonly sparseGridBuffer: GPUBuffer;
    readonly gridMassBuffer: GPUBuffer;
    readonly gridMomentumXBuffer: GPUBuffer;
    readonly gridMomentumYBuffer: GPUBuffer;
    readonly gridMomentumZBuffer: GPUBuffer;
    readonly activeBlockDispatchBuffer: GPUBuffer;

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
        const sparseGridBuffer = device.createBuffer({
            label: "MPM sparse grid storage buffer",
            size: sparseGridBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const gridStorageSize = this.nMaxActiveBlocks * GRID_CELLS_PER_BLOCK * 4;
        const gridMassBuffer = device.createBuffer({
            label: "MPM physical mass buffer",
            size: gridStorageSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const gridMomentumXBuffer = device.createBuffer({
            label: "MPM physical momentum X buffer",
            size: gridStorageSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const gridMomentumYBuffer = device.createBuffer({
            label: "MPM physical momentum Y buffer",
            size: gridStorageSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const gridMomentumZBuffer = device.createBuffer({
            label: "MPM physical momentum Z buffer",
            size: gridStorageSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // 0..2: active block dispatch, 3: active block count,
        // 4..6: clear-cell dispatch, 7: active cell count.
        const activeBlockDispatchBuffer = device.createBuffer({
            label: "MPM active block indirect dispatch buffer",
            size: 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        });

        this.particleDataBuffer = particleDataBuffer;
        this.sparseGridBuffer = sparseGridBuffer;
        this.gridMassBuffer = gridMassBuffer;
        this.gridMomentumXBuffer = gridMomentumXBuffer;
        this.gridMomentumYBuffer = gridMomentumYBuffer;
        this.gridMomentumZBuffer = gridMomentumZBuffer;
        this.activeBlockDispatchBuffer = activeBlockDispatchBuffer;

        this.nParticles = nParticles;
    }
}
