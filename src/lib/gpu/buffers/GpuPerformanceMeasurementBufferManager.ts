export class GpuPerformanceMeasurementBufferManager {
    readonly querySet: GPUQuerySet;

    readonly resolveComputeBuffer: GPUBuffer;
    readonly resultComputeBuffer: GPUBuffer;

    readonly resolveRenderBuffer: GPUBuffer;
    readonly resultRenderBuffer: GPUBuffer;

    constructor({
        device,
    }: {
        device: GPUDevice,
    }) {
        const querySet = device.createQuerySet({
            type: "timestamp",
            count: 2,
        });



        const resolveComputeBuffer = device.createBuffer({
            size: querySet.count * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });

        const resultComputeBuffer = device.createBuffer({
            size: resolveComputeBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });



        const resolveRenderBuffer = device.createBuffer({
            size: querySet.count * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });

        const resultRenderBuffer = device.createBuffer({
            size: resolveRenderBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });



        this.querySet = querySet;

        this.resolveComputeBuffer = resolveComputeBuffer;
        this.resultComputeBuffer = resultComputeBuffer;

        this.resolveRenderBuffer = resolveRenderBuffer;
        this.resultRenderBuffer = resultRenderBuffer;
    }
    
    addComputeResolve(commandEncoder: GPUCommandEncoder) {
        if (this.resultRenderBuffer.mapState !== "unmapped") return;

        commandEncoder.resolveQuerySet(this.querySet, 0, this.querySet.count, this.resolveComputeBuffer, 0);
        commandEncoder.copyBufferToBuffer(this.resolveComputeBuffer, this.resultComputeBuffer);
    }

    addRenderResolve(commandEncoder: GPUCommandEncoder) {
        if (this.resultRenderBuffer.mapState !== "unmapped") return;

        commandEncoder.resolveQuerySet(this.querySet, 0, this.querySet.count, this.resolveRenderBuffer, 0);
        commandEncoder.copyBufferToBuffer(this.resolveRenderBuffer, this.resultRenderBuffer);
    }

    private async mapResultBuffer(buffer: GPUBuffer) {
        if (buffer.mapState === "pending") return null;
        
        await buffer.mapAsync(GPUMapMode.READ);

        const startEndGpuTimestamps = new BigUint64Array(buffer.getMappedRange());
        const gpuElapsedTimeNs = startEndGpuTimestamps[1] - startEndGpuTimestamps[0];

        buffer.unmap();

        return gpuElapsedTimeNs;
    }


    async mapGpuElapsedComputeTimeNs() {
        return this.mapResultBuffer(this.resultComputeBuffer);
    }

    async mapGpuElapsedRenderTimeNs() {
        return this.mapResultBuffer(this.resultRenderBuffer);
    }
}