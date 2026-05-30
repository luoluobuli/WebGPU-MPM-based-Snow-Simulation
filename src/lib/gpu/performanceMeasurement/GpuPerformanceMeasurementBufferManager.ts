export class GpuPerformanceMeasurementBufferManager {
    readonly querySet: GPUQuerySet;

    readonly resolveBuffer: GPUBuffer;
    readonly resultBuffer: GPUBuffer;
    private isMappingResultBuffer = false;

    constructor({
        device,
    }: {
        device: GPUDevice,
    }) {
        const querySet = device.createQuerySet({
            type: "timestamp",
            count: 16,
        });


        const resolveRenderBuffer = device.createBuffer({
            size: querySet.count * 32,
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
        if (this.isMappingResultBuffer || this.resultBuffer.mapState !== "unmapped") return;

        commandEncoder.resolveQuerySet(this.querySet, 0, this.querySet.count, this.resolveBuffer, 0);
        commandEncoder.copyBufferToBuffer(this.resolveBuffer, 0, this.resultBuffer, 0, this.resolveBuffer.size);
    }


    async mapTime(fn: (timestamps: BigUint64Array) => void) {
        if (this.isMappingResultBuffer || this.resultBuffer.mapState !== "unmapped") return null;

        this.isMappingResultBuffer = true;
        let mapped = false;
        try {
            await this.resultBuffer.mapAsync(GPUMapMode.READ);
            mapped = true;

            const startEndGpuTimestamps = new BigUint64Array(this.resultBuffer.getMappedRange());
            fn(startEndGpuTimestamps);
        } finally {
            if (mapped) {
                this.resultBuffer.unmap();
            }
            this.isMappingResultBuffer = false;
        }
    }

    buildPrerenderTimestampWritesDescriptor(prerenderPassIndex: number) {
        return {
            querySet: this.querySet,
            beginningOfPassWriteIndex: 2 * (prerenderPassIndex + 3),
            endOfPassWriteIndex: 2 * (prerenderPassIndex + 3) + 1,
        };
    }
}
