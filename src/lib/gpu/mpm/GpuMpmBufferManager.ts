
export class GpuMpmBufferManager {
    // Particle data
    readonly particleDataBuffer: GPUBuffer;
    readonly sortedParticleIndicesBuffer: GPUBuffer;
    readonly nParticles: number;

    // Dense grid buffers (replaces sparse hash map)
    readonly gridMassBuffer: GPUBuffer;
    readonly gridMomentumXBuffer: GPUBuffer;
    readonly gridMomentumYBuffer: GPUBuffer;
    readonly gridMomentumZBuffer: GPUBuffer;
    readonly totalGridCells: number;

    // Bukkit system buffers
    readonly bukkitCountBuffer: GPUBuffer;
    readonly bukkitThreadDataBuffer: GPUBuffer;
    readonly bukkitParticleAllocatorBuffer: GPUBuffer;
    readonly bukkitIndexStartBuffer: GPUBuffer;
    readonly bukkitDispatchBuffer: GPUBuffer;
    readonly bukkitInsertCountersBuffer: GPUBuffer;

    readonly BUKKIT_SIZE = 4;
    readonly bukkitCountX: number;
    readonly bukkitCountY: number;
    readonly bukkitCountZ: number;
    readonly totalBukkits: number;

    // Bukkit params uniform buffer
    readonly bukkitParamsBuffer: GPUBuffer;

    constructor({
        device,
        nParticles,
        gridResolutionX,
        gridResolutionY,
        gridResolutionZ,
    }: {
        device: GPUDevice,
        nParticles: number,
        gridResolutionX: number,
        gridResolutionY: number,
        gridResolutionZ: number,
    }) {
        this.nParticles = nParticles;

        // Calculate bukkit dimensions
        this.bukkitCountX = Math.ceil(gridResolutionX / this.BUKKIT_SIZE);
        this.bukkitCountY = Math.ceil(gridResolutionY / this.BUKKIT_SIZE);
        this.bukkitCountZ = Math.ceil(gridResolutionZ / this.BUKKIT_SIZE);
        this.totalBukkits = this.bukkitCountX * this.bukkitCountY * this.bukkitCountZ;
        this.totalGridCells = gridResolutionX * gridResolutionY * gridResolutionZ;

        // Particle data buffer (192 bytes per particle)
        this.particleDataBuffer = device.createBuffer({
            label: "MPM particle data buffer",
            size: nParticles * 192,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
        });

        // Sorted particle indices buffer
        this.sortedParticleIndicesBuffer = device.createBuffer({
            label: "MPM sorted particle indices buffer",
            size: nParticles * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Dense grid buffers - O(1) access, no hash map!
        this.gridMassBuffer = device.createBuffer({
            label: "MPM dense grid mass buffer",
            size: this.totalGridCells * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.gridMomentumXBuffer = device.createBuffer({
            label: "MPM dense grid momentum X buffer",
            size: this.totalGridCells * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.gridMomentumYBuffer = device.createBuffer({
            label: "MPM dense grid momentum Y buffer",
            size: this.totalGridCells * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.gridMomentumZBuffer = device.createBuffer({
            label: "MPM dense grid momentum Z buffer",
            size: this.totalGridCells * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Bukkit system buffers
        this.bukkitCountBuffer = device.createBuffer({
            label: "MPM bukkit count buffer",
            size: this.totalBukkits * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // BukkitThreadData: 6 u32s = 24 bytes per bukkit dispatch
        // Allow up to 10x bukkits for multiple dispatches per bukkit
        this.bukkitThreadDataBuffer = device.createBuffer({
            label: "MPM bukkit thread data buffer",
            size: this.totalBukkits * 10 * 24,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Single u32 for particle allocation counter
        this.bukkitParticleAllocatorBuffer = device.createBuffer({
            label: "MPM bukkit particle allocator buffer",
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Start index per bukkit
        this.bukkitIndexStartBuffer = device.createBuffer({
            label: "MPM bukkit index start buffer",
            size: this.totalBukkits * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Indirect dispatch buffer: [dispatchX, dispatchY, dispatchZ]
        this.bukkitDispatchBuffer = device.createBuffer({
            label: "MPM bukkit dispatch buffer",
            size: 12,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        });

        // Insert counters per bukkit (for atomic insertion)
        this.bukkitInsertCountersBuffer = device.createBuffer({
            label: "MPM bukkit insert counters buffer",
            size: this.totalBukkits * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Bukkit params uniform buffer: countX, countY, countZ, particleCount
        this.bukkitParamsBuffer = device.createBuffer({
            label: "MPM bukkit params uniform buffer",
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Initialize bukkit params
        const bukkitParamsData = new Uint32Array([
            this.bukkitCountX,
            this.bukkitCountY,
            this.bukkitCountZ,
            nParticles,
        ]);
        device.queue.writeBuffer(this.bukkitParamsBuffer, 0, bukkitParamsData);
    }

    // Initialize dispatch buffer with [0, 1, 1] for indirect dispatch
    initializeDispatchBuffer(device: GPUDevice) {
        const dispatchData = new Uint32Array([0, 1, 1]);
        device.queue.writeBuffer(this.bukkitDispatchBuffer, 0, dispatchData);
    }
}