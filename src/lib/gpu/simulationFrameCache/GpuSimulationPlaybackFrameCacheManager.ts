import simulationPlaybackFrameCacheSrc from "./simulationPlaybackFrameCache.wgsl?raw";

export const SIMULATION_PLAYBACK_FRAME_BYTES_PER_PARTICLE = 16;

export type SimulationPlaybackFrameLayout = {
    positionByteLength: number,
    materialByteLength: number,
    byteLength: number,
};

const WORKGROUP_SIZE = 256;

export class GpuSimulationPlaybackFrameCacheManager {
    readonly frameBuffer: GPUBuffer;
    readonly frameByteLength: number;

    private readonly device: GPUDevice;
    private readonly nWorkgroups: number;
    private readonly packBindGroup: GPUBindGroup;
    private readonly restoreBindGroup: GPUBindGroup;
    private readonly packPipeline: GPUComputePipeline;
    private readonly restorePipeline: GPUComputePipeline;
    private destroyed = false;

    constructor({
        device,
        nParticles,
        particleDataBuffer,
        particleFlagsBuffer,
    }: {
        device: GPUDevice,
        nParticles: number,
        particleDataBuffer: GPUBuffer,
        particleFlagsBuffer: GPUBuffer,
    }) {
        const frameByteLength = Math.max(
            SIMULATION_PLAYBACK_FRAME_BYTES_PER_PARTICLE * nParticles,
            4,
        );
        const frameBuffer = device.createBuffer({
            label: "simulation playback frame buffer",
            size: frameByteLength,
            usage:
                GPUBufferUsage.STORAGE
                | GPUBufferUsage.COPY_SRC
                | GPUBufferUsage.COPY_DST,
        });
        const packBindGroupLayout = device.createBindGroupLayout({
            label: "simulation playback frame cache bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
            ],
        });
        const restoreBindGroupLayout = device.createBindGroupLayout({
            label: "simulation playback frame restore bind group layout",
            entries: [
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
            ],
        });
        const packBindGroup = device.createBindGroup({
            label: "simulation playback frame cache pack bind group",
            layout: packBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: particleDataBuffer },
                },
                {
                    binding: 1,
                    resource: { buffer: particleFlagsBuffer },
                },
                {
                    binding: 2,
                    resource: { buffer: frameBuffer },
                },
            ],
        });
        const restoreBindGroup = device.createBindGroup({
            label: "simulation playback frame cache restore bind group",
            layout: restoreBindGroupLayout,
            entries: [
                {
                    binding: 2,
                    resource: { buffer: frameBuffer },
                },
                {
                    binding: 3,
                    resource: { buffer: particleDataBuffer },
                },
                {
                    binding: 4,
                    resource: { buffer: particleFlagsBuffer },
                },
            ],
        });
        const packPipelineLayout = device.createPipelineLayout({
            label: "simulation playback frame cache pack pipeline layout",
            bindGroupLayouts: [packBindGroupLayout],
        });
        const restorePipelineLayout = device.createPipelineLayout({
            label: "simulation playback frame cache restore pipeline layout",
            bindGroupLayouts: [restoreBindGroupLayout],
        });
        const shaderModule = device.createShaderModule({
            label: "simulation playback frame cache shader module",
            code: simulationPlaybackFrameCacheSrc,
        });

        this.device = device;
        this.frameBuffer = frameBuffer;
        this.frameByteLength = frameByteLength;
        this.nWorkgroups = Math.ceil(nParticles / WORKGROUP_SIZE);
        this.packBindGroup = packBindGroup;
        this.restoreBindGroup = restoreBindGroup;
        this.packPipeline = device.createComputePipeline({
            label: "simulation playback frame pack pipeline",
            layout: packPipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: "packSimulationPlaybackFrame",
            },
        });
        this.restorePipeline = device.createComputePipeline({
            label: "simulation playback frame restore pipeline",
            layout: restorePipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: "restoreSimulationPlaybackFrame",
            },
        });
    }

    get layout(): SimulationPlaybackFrameLayout {
        return {
            positionByteLength: this.frameByteLength / 4 * 3,
            materialByteLength: this.frameByteLength / 4,
            byteLength: this.frameByteLength,
        };
    }

    addPackDispatch({
        commandEncoder,
    }: {
        commandEncoder: GPUCommandEncoder,
    }) {
        if (this.destroyed) return;

        const computePassEncoder = commandEncoder.beginComputePass({
            label: "simulation playback frame pack compute pass",
        });

        computePassEncoder.setPipeline(this.packPipeline);
        computePassEncoder.setBindGroup(0, this.packBindGroup);
        computePassEncoder.dispatchWorkgroups(this.nWorkgroups);
        computePassEncoder.end();
    }

    addRestoreDispatch({
        commandEncoder,
    }: {
        commandEncoder: GPUCommandEncoder,
    }) {
        if (this.destroyed) return;

        const computePassEncoder = commandEncoder.beginComputePass({
            label: "simulation playback frame restore compute pass",
        });

        computePassEncoder.setPipeline(this.restorePipeline);
        computePassEncoder.setBindGroup(0, this.restoreBindGroup);
        computePassEncoder.dispatchWorkgroups(this.nWorkgroups);
        computePassEncoder.end();
    }

    writeFrame(frame: ArrayBuffer) {
        if (this.destroyed) return;
        if (frame.byteLength !== this.frameByteLength) {
            throw new Error(
                `simulation playback frame has ${frame.byteLength} bytes; expected ${this.frameByteLength}`,
            );
        }

        this.device.queue.writeBuffer(
            this.frameBuffer,
            0,
            frame,
            0,
            frame.byteLength,
        );
    }

    destroy() {
        if (this.destroyed) return;

        this.destroyed = true;
        this.frameBuffer.destroy();
    }
}
