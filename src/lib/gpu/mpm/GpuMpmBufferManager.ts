
export class GpuMpmBufferManager {
    readonly particleDataBuffer: GPUBuffer;
    readonly sparseGridBuffer: GPUBuffer;
    readonly gridMassBuffer: GPUBuffer;
    readonly gridMomentumXBuffer: GPUBuffer;
    readonly gridMomentumYBuffer: GPUBuffer;
    readonly gridMomentumZBuffer: GPUBuffer;
    readonly sortedParticleIndicesBuffer: GPUBuffer;

    readonly nParticles: number;
    readonly nMaxBlocksInHashMap: number = 100_000;
    readonly hashMapSize: number = 200_003;

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
        // - n_allocated_blocks: atomic<u32> (4 bytes) + 12 bytes padding = 16 bytes
        // - hash_map_entries: array<HashMapEntry, HASH_MAP_SIZE> = HASH_MAP_SIZE * 16 bytes
        // - mapped_block_indexes: array<u32, N_MAX_BLOCKS_IN_HASH_MAP> = N_MAX_BLOCKS * 4 bytes
        // - block_particle_counts: array<u32, N_MAX_BLOCKS_IN_HASH_MAP> = N_MAX_BLOCKS * 4 bytes
        // - block_particle_offsets: array<u32, N_MAX_BLOCKS_IN_HASH_MAP> = N_MAX_BLOCKS * 4 bytes
        const sparseGridBufferSize = 16 + this.hashMapSize * 16 + this.nMaxBlocksInHashMap * 4 * 3;
        const sparseGridBuffer = device.createBuffer({
            label: "MPM sparse grid storage buffer",
            size: sparseGridBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const poolSize = this.nMaxBlocksInHashMap * 64 * 4;

        const gridMassBuffer = device.createBuffer({
            label: "MPM physical mass buffer",
            size: poolSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const gridMomentumXBuffer = device.createBuffer({
            label: "MPM physical momentum X buffer",
            size: poolSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const gridMomentumYBuffer = device.createBuffer({
            label: "MPM physical momentum Y buffer",
            size: poolSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const gridMomentumZBuffer = device.createBuffer({
            label: "MPM physical momentum Z buffer",
            size: poolSize,
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