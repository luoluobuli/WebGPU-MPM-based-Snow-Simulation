
const BLOCK_SIZE = 4;
const BUKKIT_DOMAIN_GRID_RESOLUTION = 384;
const BUKKIT_DOMAIN_BLOCKS_PER_AXIS = Math.ceil(BUKKIT_DOMAIN_GRID_RESOLUTION / BLOCK_SIZE);
const BUKKIT_DOMAIN_BLOCK_COUNT = BUKKIT_DOMAIN_BLOCKS_PER_AXIS ** 3;
const N_MAX_ACTIVE_BLOCKS = 100_000;
const GRID_CELLS_PER_BLOCK = 64;

export class GpuMpmBufferManager {
    readonly particleDataBuffer: GPUBuffer;
    readonly sparseGridBuffer: GPUBuffer;
    readonly gridMassBuffer: GPUBuffer;
    readonly gridMomentumXBuffer: GPUBuffer;
    readonly gridMomentumYBuffer: GPUBuffer;
    readonly gridMomentumZBuffer: GPUBuffer;
    readonly sortedParticleIndicesBuffer: GPUBuffer;

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
        // - n_allocated_blocks: atomic<u32> (4 bytes) + 12 bytes explicit padding = 16 bytes
        // - block_index_bukkits: array<atomic<u32>, BUKKIT_DOMAIN_BLOCK_COUNT>
        // - mapped_block_numbers: array<vec4i, N_MAX_ACTIVE_BLOCKS>
        // - block_particle_counts: array<u32, N_MAX_ACTIVE_BLOCKS>
        // - block_particle_offsets: array<u32, N_MAX_ACTIVE_BLOCKS>
        const sparseGridBufferSize = 16
            + this.bukkitDomainBlockCount * 4
            + this.nMaxActiveBlocks * 16
            + this.nMaxActiveBlocks * 4 * 2;
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

        const sortedParticleIndicesBuffer = device.createBuffer({
            label: "MPM sorted particle indices buffer",
            size: nParticles * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.particleDataBuffer = particleDataBuffer;
        this.sparseGridBuffer = sparseGridBuffer;
        this.gridMassBuffer = gridMassBuffer;
        this.gridMomentumXBuffer = gridMomentumXBuffer;
        this.gridMomentumYBuffer = gridMomentumYBuffer;
        this.gridMomentumZBuffer = gridMomentumZBuffer;
        this.sortedParticleIndicesBuffer = sortedParticleIndicesBuffer;

        this.nParticles = nParticles;
    }
}
