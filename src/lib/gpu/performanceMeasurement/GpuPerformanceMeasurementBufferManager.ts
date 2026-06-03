export class GpuPerformanceMeasurementBufferManager {
    readonly querySet: GPUQuerySet;

    readonly resolveBuffer: GPUBuffer;
    readonly resultBuffer: GPUBuffer;
    private isMappingResultBuffer = false;
    private hasPendingReadback = false;
    private prerenderTimestampBaseIndex = 6;
    enabled = true;

    constructor({
        device,
    }: {
        device: GPUDevice,
    }) {
        const querySet = device.createQuerySet({
            type: "timestamp",
            count: 16,
        });
        const queryResultByteLength = querySet.count * BigUint64Array.BYTES_PER_ELEMENT;


        const resolveRenderBuffer = device.createBuffer({
            size: queryResultByteLength,
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

    setEnabled(enabled: boolean) {
        this.enabled = enabled;
    }

    setPrerenderTimestampBaseIndex(baseIndex: number) {
        this.prerenderTimestampBaseIndex = baseIndex;
    }
    

    addResolve(commandEncoder: GPUCommandEncoder, queryCount = this.querySet.count) {
        if (!this.enabled || this.isMappingResultBuffer || this.resultBuffer.mapState !== "unmapped") return;

        const resolvedQueryCount = Math.min(this.querySet.count, Math.max(0, queryCount));
        if (resolvedQueryCount === 0) return;

        const resolvedByteLength = resolvedQueryCount * BigUint64Array.BYTES_PER_ELEMENT;
        commandEncoder.resolveQuerySet(this.querySet, 0, resolvedQueryCount, this.resolveBuffer, 0);
        commandEncoder.copyBufferToBuffer(this.resolveBuffer, 0, this.resultBuffer, 0, resolvedByteLength);
        this.hasPendingReadback = true;
    }


    async mapTime(fn: (timestamps: BigUint64Array) => void) {
        if (!this.hasPendingReadback || this.isMappingResultBuffer || this.resultBuffer.mapState !== "unmapped") {
            return null;
        }

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
            this.hasPendingReadback = false;
            this.isMappingResultBuffer = false;
        }
    }

    buildPrerenderTimestampWritesDescriptor(prerenderPassIndex: number) {
        if (!this.enabled) return undefined;

        return {
            querySet: this.querySet,
            beginningOfPassWriteIndex: this.prerenderTimestampBaseIndex + 2 * prerenderPassIndex,
            endOfPassWriteIndex: this.prerenderTimestampBaseIndex + 2 * prerenderPassIndex + 1,
        };
    }
}
