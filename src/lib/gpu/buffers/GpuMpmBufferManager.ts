

export class GpuMpmBufferManager {
    readonly particleDataBuffer: GPUBuffer;
    readonly gridDataBuffer: GPUBuffer;


    constructor({
        device,
        nParticles,
        gridResolution,
        initialPositions = null,
    }: {
        device: GPUDevice,
        nParticles: number,
        gridResolution: number,
        initialPositions?: Float32Array | null,
    }) {
        const particleDataBuffer = device.createBuffer({
            label: "particle data buffer",
            size: nParticles * 48,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
        });
        const particleDataArray = new Float32Array(nParticles * 12);
        
        if (initialPositions !== null && initialPositions.length >= nParticles * 3) {
            for (let i = 0; i < nParticles; i++) {
                const offset = i * 12;

                particleDataArray.set(
                    new Float32Array([
                        // pos
                        initialPositions[i * 3], initialPositions[i * 3 + 1], initialPositions[i * 3 + 2], 1,
                        // vel
                        0, 0, 0, 0,
                        // affine + mass
                        0, 0, 0, 1,
                    ]),
                    offset,
                );
            }
        } else {
            for (let i = 0; i < nParticles; i++) {
                const offset = i * 12;

                particleDataArray.set(
                    new Float32Array([
                        // pos
                        Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1, 1,
                        // vel
                        0, 0, 0, 0,
                        // affine + mass
                        0, 0, 0, 1,
                    ]),
                    offset,
                );
            }
        }
        device.queue.writeBuffer(particleDataBuffer, 0, particleDataArray);


        const gridDataBuffer = device.createBuffer({
            label: "grid data buffer",
            size: (gridResolution**3) * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });


        this.particleDataBuffer = particleDataBuffer;

        this.gridDataBuffer = gridDataBuffer;
    }
}