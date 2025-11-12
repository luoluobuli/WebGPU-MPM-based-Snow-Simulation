import type { GpuUniformsBufferManager } from "../buffers/GpuUniformsBufferManager";
import commonModuleSrc from "../shaders/_common.wgsl?raw";
import simulationStepModuleSrc from "../shaders/simulationStep.wgsl?raw";
import p2gModuleSrc from "../shaders/particleToGrid.cs.wgsl?raw";
import gridUpdateModuleSrc from "../shaders/gridUpdate.cs.wgsl?raw";
import g2pModuleSrc from "../shaders/gridToParticle.cs.wgsl?raw";

export class GpuSimulationStepPipelineManager {
    readonly storageBindGroupLayout: GPUBindGroupLayout;
    readonly storageBindGroup: GPUBindGroup;

    //readonly computePipeline: GPUComputePipeline;
    readonly p2gComputePipeline: GPUComputePipeline;
    readonly gridComputePipeline: GPUComputePipeline;
    readonly g2pComputePipeline: GPUComputePipeline;

    private readonly uniformsManager: GpuUniformsBufferManager;

    constructor({
        device,
        particleDataBuffer1,
        particleDataBuffer2,
        gridDataBuffer1,
        gridDataBuffer2,
        uniformsManager,
    }: {
        device: GPUDevice,
        particleDataBuffer1: GPUBuffer,
        particleDataBuffer2: GPUBuffer,
        gridDataBuffer1: GPUBuffer,
        gridDataBuffer2: GPUBuffer,
        uniformsManager: GpuUniformsBufferManager,
    }) {
        const simulationStepStorageBindGroupLayout = device.createBindGroupLayout({
            label: "simulation step storage bind group layout",
            
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
                        type: "storage",
                    },
                },
            ],
        });
        
        const simulationStepStorageBindGroup = device.createBindGroup({
            label: "simulation step storage bind group",

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
                        buffer: gridDataBuffer1,
                    },
                },
            ],
        });

        const simulationStepPipelineLayout = device.createPipelineLayout({
            label: "simulation step pipeline layout",
            bindGroupLayouts: [uniformsManager.bindGroupLayout, simulationStepStorageBindGroupLayout],
        });

        // Load shader modules
        // const simulationStepModule = device.createShaderModule({
        //     label: "simulation step module",
        //     code: commonModuleSrc + simulationStepModuleSrc,
        // });
        const p2gModule = device.createShaderModule({
            code: commonModuleSrc + p2gModuleSrc,
        });
        const gridUpdateModule = device.createShaderModule({
            code: commonModuleSrc + gridUpdateModuleSrc,
        });
        const g2pModule = device.createShaderModule({
            code: commonModuleSrc + g2pModuleSrc,
        });
        
        // Create compute pipelines
        // const simulationStepPipeline = device.createComputePipeline({
        //     label: "simulation step pipeline",
        //     layout: simulationStepPipelineLayout,

        //     compute: {
        //         module: simulationStepModule,
        //         entryPoint: "doSimulationStep",
        //     },
        // });

        this.p2gComputePipeline = device.createComputePipeline({
            label: "particle to grid compute pipeline",
            layout: simulationStepPipelineLayout,

            compute: {
                module: p2gModule,
                entryPoint: "doParticleToGrid",
            },
        });

        this.gridComputePipeline = device.createComputePipeline({
            label: "grid update compute pipeline",
            layout: simulationStepPipelineLayout,

            compute: {
                module: gridUpdateModule,
                entryPoint: "doGridUpdate",
            },
        });

        this.g2pComputePipeline = device.createComputePipeline({
            label: "grid to particle compute pipeline",
            layout: simulationStepPipelineLayout,

            compute: {
                module: g2pModule,
                entryPoint: "doGridToParticle",
            },
        });


        this.storageBindGroupLayout = simulationStepStorageBindGroupLayout;
        this.storageBindGroup = simulationStepStorageBindGroup;
        // this.storageBindGroup2_1 = simulationStepStorageBindGroup2_1;

        // this.computePipeline = simulationStepPipeline;

        this.uniformsManager = uniformsManager;
    }

    addComputePass({
        commandEncoder,
        numThreads,
        buffer1IsSource,
        pipeline,
        label,
    }: {
        commandEncoder: GPUCommandEncoder,
        numThreads: number,
        buffer1IsSource: boolean,
        pipeline: GPUComputePipeline,
        label: string
    }) {
        const computePassEncoder = commandEncoder.beginComputePass({
            label: label,
        });
        computePassEncoder.setPipeline(pipeline);
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        //computePassEncoder.setBindGroup(1, this.storageBindGroupCurrent(buffer1IsSource));
        computePassEncoder.setBindGroup(1, this.storageBindGroup);
        computePassEncoder.dispatchWorkgroups(Math.ceil(numThreads / 256));
        computePassEncoder.end();
    }

    // storageBindGroupCurrent(buffer1IsSource: boolean) {
    //     return buffer1IsSource
    //         ? this.storageBindGroup1_2
    //         : this.storageBindGroup2_1;
    // }
}