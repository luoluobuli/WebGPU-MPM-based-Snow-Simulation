import type { GpuUniformsBufferManager } from "../buffers/GpuUniformsBufferManager";
import commonModuleSrc from "../shaders/_common.wgsl?raw";
import particleScatterModuleSrc from "../shaders/particleScatter.cs.wgsl?raw";

export class GpuParticleInitPipelineManager {
    readonly storageBindGroupLayout: GPUBindGroupLayout;
    readonly storageBindGroup: GPUBindGroup;
    readonly computePipeline: GPUComputePipeline;

    private readonly uniformsManager: GpuUniformsBufferManager;

    constructor({
        device,
        particleDataBuffer,
        meshVerticesBuffer,
        uniformsManager,
    }: {
        device: GPUDevice,
        particleDataBuffer: GPUBuffer,
        meshVerticesBuffer: GPUBuffer,
        uniformsManager: GpuUniformsBufferManager,
    }) {
        const storageBindGroupLayout = device.createBindGroupLayout({
            label: "particle scatter storage bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage",
                    },
                },
            ],
        });

        const storageBindGroup = device.createBindGroup({
            label: "particle scatter storage bind group",
            layout: storageBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: particleDataBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: meshVerticesBuffer,
                    },
                },
            ],
        });

        const pipelineLayout = device.createPipelineLayout({
            label: "particle scatter pipeline layout",
            bindGroupLayouts: [
                uniformsManager.bindGroupLayout,
                storageBindGroupLayout,
            ],
        });

        const shaderModule = device.createShaderModule({
            code: commonModuleSrc + particleScatterModuleSrc,
        });

        this.computePipeline = device.createComputePipeline({
            label: "particle scatter compute pipeline",
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: "scatterParticles",
            },
        });

        this.storageBindGroupLayout = storageBindGroupLayout;
        this.storageBindGroup = storageBindGroup;
        this.uniformsManager = uniformsManager;
    }

    addDispatch({
        commandEncoder,
        nParticles,
    }: {
        commandEncoder: GPUCommandEncoder,
        nParticles: number,
    }) {
        const computePassEncoder = commandEncoder.beginComputePass({
            label: "particle init compute pass",
        });

        computePassEncoder.setPipeline(this.computePipeline);
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.storageBindGroup);
        computePassEncoder.dispatchWorkgroups(Math.ceil(nParticles / 256));

        computePassEncoder.end();
    }
}
