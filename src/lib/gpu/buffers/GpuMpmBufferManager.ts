

export class GpuMpmBufferManager {
    readonly particleDataBuffer: GPUBuffer;
    readonly gridDataBuffer: GPUBuffer;
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
            label: "particle data buffer",
            size: nParticles * 128,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
        });


        const gridDataBuffer = device.createBuffer({
            label: "grid data buffer",
            size: gridResolutionX * gridResolutionY * gridResolutionZ * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });


        this.particleDataBuffer = particleDataBuffer;
        this.gridDataBuffer = gridDataBuffer;
        this.nParticles = nParticles;
    }
}