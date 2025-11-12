

export class GpuMpmBufferManager {
    readonly particleDataBuffer1: GPUBuffer;
    readonly particleDataBuffer2: GPUBuffer;
    readonly gridDataBuffer1: GPUBuffer;
    readonly gridDataBuffer2: GPUBuffer;


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
        const particleDataBuffer1 = device.createBuffer({
            label: "particle data ping-pong buffer 1",
            size: nParticles * 80,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
        });
        const particleDataBuffer2 = device.createBuffer({
            label: "particle data ping-pong buffer 2",
            size: nParticles * 80,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
        });
        const particleDataArray = new Float32Array(nParticles * 20);
        
        if (initialPositions !== null && initialPositions.length >= nParticles * 3) {
            for (let i = 0; i < nParticles; i++) {
                const offset = i * 20;

                particleDataArray.set(
                    new Float32Array([
                        // pos
                        initialPositions[i * 3], initialPositions[i * 3 + 1], initialPositions[i * 3 + 2], 1,
                        // vel
                        0, 0, 0, 0,
                        // deform
                        1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                    ]),
                    offset,
                );
            }
        } else {
            for (let i = 0; i < nParticles; i++) {
                const offset = i * 20;

                particleDataArray.set(
                    new Float32Array([
                        // pos
                        Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1, 1,
                        // vel
                        0, 0, 0, 0,
                        // deform
                        1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                    ]),
                    offset,
                );
            }
        }
        device.queue.writeBuffer(particleDataBuffer1, 0, particleDataArray);


        const gridDataBuffer1 = device.createBuffer({
            label: "grid data ping-pong buffer 1",
            size: (gridResolution**3) * 16,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
        });
        const gridDataBuffer2 = device.createBuffer({
            label: "grid data ping-pong buffer 2",
            size: (gridResolution**3) * 16,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
        });


        this.particleDataBuffer1 = particleDataBuffer1;
        this.particleDataBuffer2 = particleDataBuffer2;

        this.gridDataBuffer1 = gridDataBuffer1;
        this.gridDataBuffer2 = gridDataBuffer2;
    }

    particleDataBufferCurrent(buffer1IsSource: boolean) {
        return buffer1IsSource
            ? this.particleDataBuffer1
            : this.particleDataBuffer2;
    }

    gridDataBufferCurrent(buffer1IsSource: boolean) {
        return buffer1IsSource
            ? this.gridDataBuffer1
            : this.gridDataBuffer2;
    }
}