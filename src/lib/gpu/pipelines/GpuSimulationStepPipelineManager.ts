import type { GpuUniformsBufferManager } from "../buffers/GpuUniformsBufferManager";
import p2gModuleSrc from "../shaders/particleToGrid.cs.wgsl?raw";
import gridUpdateModuleSrc from "../shaders/gridUpdate.cs.wgsl?raw";
import g2pModuleSrc from "../shaders/gridToParticle.cs.wgsl?raw";
import gridClearModuleSrc from "../shaders/gridClear.cs.wgsl?raw";
import { attachPrelude } from "../shaders/prelude";

export class GpuSimulationStepPipelineManager {
    readonly storageBindGroupLayout: GPUBindGroupLayout;
    readonly storageBindGroup: GPUBindGroup;

    readonly gridClearComputePipeline: GPUComputePipeline;
    readonly p2gComputePipeline: GPUComputePipeline;
    readonly gridComputePipeline: GPUComputePipeline;
    readonly g2pComputePipeline: GPUComputePipeline;

    private readonly uniformsManager: GpuUniformsBufferManager;

    constructor({
        device,
        particleDataBuffer,
        gridDataBuffer,
        uniformsManager,
    }: {
        device: GPUDevice,
        particleDataBuffer: GPUBuffer,
        gridDataBuffer: GPUBuffer,
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
                        buffer: particleDataBuffer,
                    },
                },

                {
                    binding: 1,
                    resource: {
                        buffer: gridDataBuffer,
                    },
                },
            ],
        });

        const simulationStepPipelineLayout = device.createPipelineLayout({
            label: "simulation step pipeline layout",
            bindGroupLayouts: [uniformsManager.bindGroupLayout, simulationStepStorageBindGroupLayout],
        });


        const gridClearModule = device.createShaderModule({
            code: attachPrelude(gridClearModuleSrc),
        });
        const p2gModule = device.createShaderModule({
            code: attachPrelude(p2gModuleSrc),
        });
        const gridUpdateModule = device.createShaderModule({
            code: attachPrelude(gridUpdateModuleSrc),
        });
        const g2pModule = device.createShaderModule({
            code: attachPrelude(g2pModuleSrc),
        });
        

        this.gridClearComputePipeline = device.createComputePipeline({
            label: "grid clear compute pipeline",
            layout: simulationStepPipelineLayout,

            compute: {
                module: gridClearModule,
                entryPoint: "doClearGrid",
            },
        });

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

        this.uniformsManager = uniformsManager;
    }

    addDispatch({
        computePassEncoder,
        pipeline,
        dispatchX,
        dispatchY,
        dispatchZ,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        pipeline: GPUComputePipeline,
        dispatchX : number,
        dispatchY? : number,
        dispatchZ? : number,
    }) {
        computePassEncoder.setPipeline(pipeline);
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.storageBindGroup);
        computePassEncoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    }
}