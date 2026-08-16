export type KinematicColliderIntervalWriter = (args: {
    intervalIndex: number,
    previousTimeS: number,
    currentTimeS: number,
    target: Uint8Array,
}) => void;

export class GpuKinematicColliderIntervalBatchBuffer {
    readonly intervalPoseByteLength: number;

    private readonly device: GPUDevice;
    private buffer: GPUBuffer | null = null;
    private upload = new Uint8Array(0);
    private readonly retiredBuffers: GPUBuffer[] = [];
    private intervalCapacity = 0;
    private writtenIntervalCount = 0;

    constructor({
        device,
        intervalPoseByteLength,
    }: {
        device: GPUDevice,
        intervalPoseByteLength: number,
    }) {
        if (
            !Number.isInteger(intervalPoseByteLength)
            || intervalPoseByteLength <= 0
            || intervalPoseByteLength % 4 !== 0
        ) {
            throw new Error(
                "kinematic collider interval pose byte length must be a "
                + "positive multiple of four",
            );
        }

        this.device = device;
        this.intervalPoseByteLength = intervalPoseByteLength;
    }

    private ensureCapacity(intervalCount: number) {
        if (intervalCount <= this.intervalCapacity) return;

        const totalByteLength =
            this.intervalPoseByteLength * intervalCount;
        if (!Number.isSafeInteger(totalByteLength)) {
            throw new Error(
                "kinematic collider interval batch byte length is unsafe",
            );
        }

        const upload = new Uint8Array(totalByteLength);
        const buffer = this.device.createBuffer({
            label: "kinematic collider interval pose batch buffer",
            size: totalByteLength,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        if (this.buffer !== null) {
            // A previous submission may still reference this buffer.
            this.retiredBuffers.push(this.buffer);
        }
        this.upload = upload;
        this.buffer = buffer;
        this.intervalCapacity = intervalCount;
    }

    writeIntervals({
        intervalCount,
        startTimeS,
        simulationTimestepS,
        writeInterval,
    }: {
        intervalCount: number,
        startTimeS: number,
        simulationTimestepS: number,
        writeInterval: KinematicColliderIntervalWriter,
    }) {
        if (!Number.isInteger(intervalCount) || intervalCount <= 0) {
            throw new Error(
                "kinematic collider interval count must be a positive integer",
            );
        }
        if (
            !Number.isFinite(startTimeS)
            || !Number.isFinite(simulationTimestepS)
            || simulationTimestepS <= 0
        ) {
            throw new Error(
                "kinematic collider interval timing must be finite with a "
                + "positive timestep",
            );
        }

        this.writtenIntervalCount = 0;
        this.ensureCapacity(intervalCount);
        const buffer = this.buffer!;

        for (
            let intervalIndex = 0;
            intervalIndex < intervalCount;
            intervalIndex++
        ) {
            const targetByteOffset =
                intervalIndex * this.intervalPoseByteLength;
            const previousTimeS =
                startTimeS + intervalIndex * simulationTimestepS;
            const currentTimeS = previousTimeS + simulationTimestepS;
            writeInterval({
                intervalIndex,
                previousTimeS,
                currentTimeS,
                target: this.upload.subarray(
                    targetByteOffset,
                    targetByteOffset + this.intervalPoseByteLength,
                ),
            });
        }

        const writtenByteLength =
            intervalCount * this.intervalPoseByteLength;
        this.device.queue.writeBuffer(
            buffer,
            0,
            this.upload.buffer,
            0,
            writtenByteLength,
        );
        this.writtenIntervalCount = intervalCount;
    }

    copyIntervalTo({
        commandEncoder,
        intervalIndex,
        targetBuffer,
        targetByteOffset = 0,
    }: {
        commandEncoder: GPUCommandEncoder,
        intervalIndex: number,
        targetBuffer: GPUBuffer,
        targetByteOffset?: number,
    }) {
        if (
            !Number.isInteger(intervalIndex)
            || intervalIndex < 0
            || intervalIndex >= this.writtenIntervalCount
        ) {
            throw new Error(
                "kinematic collider interval index has not been written",
            );
        }
        if (
            !Number.isInteger(targetByteOffset)
            || targetByteOffset < 0
            || targetByteOffset % 4 !== 0
        ) {
            throw new Error(
                "kinematic collider target byte offset must be a "
                + "non-negative multiple of four",
            );
        }
        if (
            targetByteOffset + this.intervalPoseByteLength
            > targetBuffer.size
        ) {
            throw new Error(
                "kinematic collider target buffer is too small",
            );
        }

        commandEncoder.copyBufferToBuffer(
            this.buffer!,
            intervalIndex * this.intervalPoseByteLength,
            targetBuffer,
            targetByteOffset,
            this.intervalPoseByteLength,
        );
    }

    destroy() {
        this.buffer?.destroy();
        for (const retiredBuffer of this.retiredBuffers) {
            retiredBuffer.destroy();
        }
    }
}