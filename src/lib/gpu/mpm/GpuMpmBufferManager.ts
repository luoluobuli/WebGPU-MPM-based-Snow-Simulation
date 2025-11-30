
export class GpuMpmBufferManager {
    readonly particleDataBuffer: GPUBuffer;
    readonly pageTableBuffer: GPUBuffer;
    readonly gridMassBuffer: GPUBuffer;
    readonly gridMomentumXBuffer: GPUBuffer;
    readonly gridMomentumYBuffer: GPUBuffer;
    readonly gridMomentumZBuffer: GPUBuffer;
    readonly nAllocatedBlocksBuffer: GPUBuffer;
    // readonly nWorkgroupsBuffer: GPUBuffer;
    readonly mappedBlockIndexesBuffer: GPUBuffer;

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
            size: nParticles * 176,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
        });


        const blocksHashMapBuffer = device.createBuffer({
            label: "MPM blocks hash map buffer",
            size: this.hashMapSize * 16,
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


        const nAllocatedBlocksBufer = device.createBuffer({
            label: "MPM # allocated blocks buffer",
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        // const nWorkgroupsBuffer = device.createBuffer({
        //     label: "MPM # workgroups buffer",
        //     size: 12,
        //     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        // });

        const mappedBlockIndexesBuffer = device.createBuffer({
            label: "MPM mapped block indexes buffer",
            size: this.nMaxBlocksInHashMap * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.particleDataBuffer = particleDataBuffer;
        this.pageTableBuffer = blocksHashMapBuffer;
        this.gridMassBuffer = gridMassBuffer;
        this.gridMomentumXBuffer = gridMomentumXBuffer;
        this.gridMomentumYBuffer = gridMomentumYBuffer;
        this.gridMomentumZBuffer = gridMomentumZBuffer;
        this.nAllocatedBlocksBuffer = nAllocatedBlocksBufer;
        // this.nWorkgroupsBuffer = nWorkgroupsBuffer;
        this.mappedBlockIndexesBuffer = mappedBlockIndexesBuffer;

        this.nParticles = nParticles;
    }
}