import type { GpuUniformsBufferManager } from "../buffers/GpuUniformsBufferManager";
import commonModuleSrc from "../shaders/_common.wgsl?raw";
import simulationStepModuleSrc from "../shaders/simulationStep.wgsl?raw";

export class GpuSimulationStepPipelineManager {
    readonly storageBindGroupLayout: GPUBindGroupLayout;
    readonly storageBindGroup1_2: GPUBindGroup;
    readonly storageBindGroup2_1: GPUBindGroup;
    readonly computePipeline: GPUComputePipeline;

    private readonly uniformsManager: GpuUniformsBufferManager;

    constructor({
        device,
        particleDataBuffer1,
        particleDataBuffer2,
        uniformsManager,
    }: {
        device: GPUDevice,
        particleDataBuffer1: GPUBuffer,
        particleDataBuffer2: GPUBuffer,
        uniformsManager: GpuUniformsBufferManager,
    }) {
        const simulationStepStorageBindGroupLayout = device.createBindGroupLayout({
            label: "simulation step storage bind group layout",
            
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage",
                    },
                },

                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
            ],
        });
        const simulationStepStorageBindGroup1_2 = device.createBindGroup({
            label: "simulation step storage bind group, 1 -> 2",

            layout: simulationStepStorageBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: particleDataBuffer1,
                    },
                },

                {
                    binding: 1,
                    resource: {
                        buffer: particleDataBuffer2,
                    },
                },
            ],
        });
        const simulationStepStorageBindGroup2_1 = device.createBindGroup({
            label: "simulation step storage bind group, 2 -> 1",

            layout: simulationStepStorageBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: particleDataBuffer2,
                    },
                },

                {
                    binding: 1,
                    resource: {
                        buffer: particleDataBuffer1,
                    },
                },
            ],
        });
        const simulationStepPipelineLayout = device.createPipelineLayout({
            label: "simulation step pipeline layout",
            bindGroupLayouts: [uniformsManager.bindGroupLayout, simulationStepStorageBindGroupLayout],
        });

        const simulationStepModule = device.createShaderModule({
            label: "simulation step module",
            code: commonModuleSrc + simulationStepModuleSrc,
        });
        
        const simulationStepPipeline = device.createComputePipeline({
            label: "simulation step pipeline",
            layout: simulationStepPipelineLayout,

            compute: {
                module: simulationStepModule,
                entryPoint: "doSimulationStep",
            },
        });


        this.storageBindGroupLayout = simulationStepStorageBindGroupLayout;
        this.storageBindGroup1_2 = simulationStepStorageBindGroup1_2;
        this.storageBindGroup2_1 = simulationStepStorageBindGroup2_1;
        this.computePipeline = simulationStepPipeline;

        this.uniformsManager = uniformsManager;
    }

    addComputePass({
        commandEncoder,
        nParticles,
        buffer1IsSource,
    }: {
        commandEncoder: GPUCommandEncoder,
        nParticles: number,
        buffer1IsSource: boolean,
    }) {
        const computePassEncoder = commandEncoder.beginComputePass({
            label: "simulation step compute pass",
        });
        computePassEncoder.setPipeline(this.computePipeline);
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.storageBindGroupCurrent(buffer1IsSource));
        computePassEncoder.dispatchWorkgroups(Math.ceil(nParticles / 256));
        computePassEncoder.end();
    }

    storageBindGroupCurrent(buffer1IsSource: boolean) {
        return buffer1IsSource
            ? this.storageBindGroup1_2
            : this.storageBindGroup2_1;
    }
}