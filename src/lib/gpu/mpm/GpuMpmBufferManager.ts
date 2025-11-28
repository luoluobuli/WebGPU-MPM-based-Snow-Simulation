

export class GpuMpmBufferManager {
    readonly particleDataBuffer: GPUBuffer;
    readonly gridMomentumXBuffer: GPUBuffer;
    readonly gridMomentumYBuffer: GPUBuffer;
    readonly gridMomentumZBuffer: GPUBuffer;
    readonly gridMassBuffer: GPUBuffer;

    readonly nParticles: number;


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
        const particleDataBuffer = device.createBuffer({
            label: "MPM particle data buffer",
            size: nParticles * 128,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
        });


        // these are split out to avoid single buffers from becoming too large

        const gridMomentumXBuffer = device.createBuffer({
            label: "MPM grid momentum X buffer",
            size: gridResolutionX * gridResolutionY * gridResolutionZ * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const gridMomentumYBuffer = device.createBuffer({
            label: "MPM grid momentum Y buffer",
            size: gridResolutionX * gridResolutionY * gridResolutionZ * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const gridMomentumZBuffer = device.createBuffer({
            label: "MPM grid momentum Z buffer",
            size: gridResolutionX * gridResolutionY * gridResolutionZ * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const gridMassBuffer = device.createBuffer({
            label: "MPM grid mass buffer",
            size: gridResolutionX * gridResolutionY * gridResolutionZ * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });


        this.particleDataBuffer = particleDataBuffer;
        this.gridMomentumXBuffer = gridMomentumXBuffer;
        this.gridMomentumYBuffer = gridMomentumYBuffer;
        this.gridMomentumZBuffer = gridMomentumZBuffer;
        this.gridMassBuffer = gridMassBuffer;

        this.nParticles = nParticles;
    }
}