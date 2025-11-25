export class GpuPerformanceMeasurementBufferManager {
    readonly querySet: GPUQuerySet;

    readonly resolveBuffer: GPUBuffer;
    readonly resultBuffer: GPUBuffer;

    constructor({
        device,
    }: {
        device: GPUDevice,
    }) {
        const querySet = device.createQuerySet({
            type: "timestamp",
            count: 2,
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

        this.resolveBuffer = resolveRenderBuffer;
        this.resultBuffer = resultRenderBuffer;
    }
    

    addResolve(commandEncoder: GPUCommandEncoder) {
        if (this.resultBuffer.mapState !== "unmapped") return;

        commandEncoder.resolveQuerySet(this.querySet, 0, this.querySet.count, this.resolveBuffer, 0);
        commandEncoder.copyBufferToBuffer(this.resolveBuffer, this.resultBuffer);
    }


    async mapTime() {
        if (this.resultBuffer.mapState === "pending") return null;
        
        await this.resultBuffer.mapAsync(GPUMapMode.READ);

        const startEndGpuTimestamps = new BigUint64Array(this.resultBuffer.getMappedRange());
        const gpuElapsedTimeNs = startEndGpuTimestamps[1] - startEndGpuTimestamps[0];

        this.resultBuffer.unmap();

        return gpuElapsedTimeNs;
    }
}