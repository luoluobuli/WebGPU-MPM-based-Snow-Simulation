const MAX_SPEED_RESULT_BUFFER_SIZE = 4;

export class GpuParticleSpeedReductionPipelineManager {
    readonly maxSpeedBuffer: GPUBuffer;
    private readonly readbackBuffer: GPUBuffer;

    private hasPendingReadback = false;
    private isMappingReadbackBuffer = false;
    private destroyed = false;

    constructor({
        device,
    }: {
        device: GPUDevice,
    }) {
        const maxSpeedBuffer = device.createBuffer({
            label: "MPM max particle speed buffer",
            size: MAX_SPEED_RESULT_BUFFER_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        const readbackBuffer = device.createBuffer({
            label: "MPM max particle speed readback buffer",
            size: MAX_SPEED_RESULT_BUFFER_SIZE,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        this.maxSpeedBuffer = maxSpeedBuffer;
        this.readbackBuffer = readbackBuffer;
    }

    reset({
        commandEncoder,
    }: {
        commandEncoder: GPUCommandEncoder,
    }) {
        if (this.destroyed) return;

        this.hasPendingReadback = false;
        commandEncoder.clearBuffer(this.maxSpeedBuffer);
    }

    canScheduleReadback() {
        return !this.destroyed
            && !this.hasPendingReadback
            && !this.isMappingReadbackBuffer
            && this.readbackBuffer.mapState === "unmapped";
    }

    copyToReadback({
        commandEncoder,
    }: {
        commandEncoder: GPUCommandEncoder,
    }) {
        if (this.destroyed) return;
        if (this.isMappingReadbackBuffer || this.readbackBuffer.mapState !== "unmapped") return;

        commandEncoder.copyBufferToBuffer(
            this.maxSpeedBuffer,
            0,
            this.readbackBuffer,
            0,
            MAX_SPEED_RESULT_BUFFER_SIZE,
        );

        this.hasPendingReadback = true;
    }

    async mapMaxSpeed(fn: (maxSpeed: number) => void) {
        if (
            this.destroyed
            || !this.hasPendingReadback
            || this.isMappingReadbackBuffer
            || this.readbackBuffer.mapState !== "unmapped"
        ) {
            return null;
        }

        this.isMappingReadbackBuffer = true;
        let mapped = false;
        try {
            await this.readbackBuffer.mapAsync(GPUMapMode.READ);
            mapped = true;

            if (this.destroyed) return null;

            const maxSpeedSquared = new Float32Array(this.readbackBuffer.getMappedRange())[0] ?? 0;
            const maxSpeed = Math.sqrt(Math.max(0, maxSpeedSquared));
            if (Number.isFinite(maxSpeed)) {
                fn(maxSpeed);
            }
        } catch (error) {
            if (this.destroyed) return null;

            throw error;
        } finally {
            const mapState = this.readbackBuffer.mapState as GPUBufferMapState;
            if (mapped && mapState === "mapped") {
                this.readbackBuffer.unmap();
            }
            this.hasPendingReadback = false;
            this.isMappingReadbackBuffer = false;
        }
    }

    destroy() {
        if (this.destroyed) return;

        this.destroyed = true;
        if (this.readbackBuffer.mapState === "mapped") {
            this.readbackBuffer.unmap();
        }
        this.maxSpeedBuffer.destroy();
        this.readbackBuffer.destroy();
    }
}
