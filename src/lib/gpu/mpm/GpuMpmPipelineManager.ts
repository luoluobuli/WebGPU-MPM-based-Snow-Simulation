import type { GpuUniformsBufferManager } from "../uniforms/GpuUniformsBufferManager";
import p2gModuleSrc from "./particleToGrid.cs.wgsl?raw";
import gridUpdateModuleSrc from "./gridUpdate.cs.wgsl?raw";
import g2pModuleSrc from "./gridToParticle.cs.wgsl?raw";
import sparseGridPreludeSrc from "./sparseGridPrelude.wgsl?raw";
import mapAffectedBlocksSrc from "./mapAffectedBlocks.wgsl?raw";
import clearHashMapSrc from "./clearHashMap.wgsl?raw";
import clearMappedBlocksSrc from "./clearMappedBlocks.wgsl?raw";
import { attachPrelude } from "../shaderPrelude";

export class GpuMpmPipelineManager {
    readonly particleBindGroupLayout: GPUBindGroupLayout;
    readonly particleDataBindGroup: GPUBindGroup;
    readonly sparseGridBindGroupLayout: GPUBindGroupLayout;
    readonly sparseGridBindGroup: GPUBindGroup;

    readonly clearHashMapPipeline: GPUComputePipeline;
    readonly mapAffectedBlocksPipeline: GPUComputePipeline;
    readonly clearMappedBlocksPipeline: GPUComputePipeline;
    readonly p2gComputePipeline: GPUComputePipeline;
    readonly gridComputePipeline: GPUComputePipeline;
    readonly g2pComputePipeline: GPUComputePipeline;

    private readonly uniformsManager: GpuUniformsBufferManager;

    constructor({
        device,
        particleDataBuffer,
        pageTableBuffer,
        gridMassBuffer,
        gridMomentumXBuffer,
        gridMomentumYBuffer,
        gridMomentumZBuffer,
        allocatorBuffer,
        // nWorkgroupsBuffer,
        mappedBlockIndexesBuffer,
        uniformsManager,
    }: {
        device: GPUDevice,
        particleDataBuffer: GPUBuffer,
        pageTableBuffer: GPUBuffer,
        gridMassBuffer: GPUBuffer,
        gridMomentumXBuffer: GPUBuffer,
        gridMomentumYBuffer: GPUBuffer,
        gridMomentumZBuffer: GPUBuffer,
        allocatorBuffer: GPUBuffer,
        // nWorkgroupsBuffer: GPUBuffer,
        mappedBlockIndexesBuffer: GPUBuffer,
        uniformsManager: GpuUniformsBufferManager,
    }) {
        const sparseGridBindGroupLayout = device.createBindGroupLayout({
            label: "MPM sparse grid bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                // { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });

        const sparseGridBindGroup = device.createBindGroup({
            label: "MPM sparse grid bind group",
            layout: sparseGridBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: pageTableBuffer } },
                { binding: 1, resource: { buffer: allocatorBuffer } },
                { binding: 2, resource: { buffer: mappedBlockIndexesBuffer } },
                { binding: 3, resource: { buffer: gridMassBuffer } },
                { binding: 4, resource: { buffer: gridMomentumXBuffer } },
                { binding: 5, resource: { buffer: gridMomentumYBuffer } },
                { binding: 6, resource: { buffer: gridMomentumZBuffer } },
                // { binding: 7, resource: { buffer: nWorkgroupsBuffer } },
            ],
        });



        const particleBindGroupLayout = device.createBindGroupLayout({
            label: "simulation step storage bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
            ],
        });
        
        const particleBindGroup = device.createBindGroup({
            label: "simulation step storage bind group",
            layout: particleBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: particleDataBuffer,
                    },
                },
            ],
        });


        const sparseGridPipelineLayout = device.createPipelineLayout({
            label: "sparse grid pipeline layout",
            bindGroupLayouts: [uniformsManager.bindGroupLayout, sparseGridBindGroupLayout],
        });

        const particlePipelineLayout = device.createPipelineLayout({
            label: "particle pipeline layout",
            bindGroupLayouts: [uniformsManager.bindGroupLayout, sparseGridBindGroupLayout, particleBindGroupLayout],
        });



        this.clearHashMapPipeline = device.createComputePipeline({
            label: "clear hash map pipeline",
            layout: sparseGridPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "clear hash map module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${clearHashMapSrc}`),
                }),
                entryPoint: "clearHashMap",
            },
        });

        this.mapAffectedBlocksPipeline = device.createComputePipeline({
            label: "map affected blocks pipeline",
            layout: particlePipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "map affected blocks module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${mapAffectedBlocksSrc}`),
                }),
                entryPoint: "mapAffectedBlocks",
            },
        });

        this.clearMappedBlocksPipeline = device.createComputePipeline({
            label: "clear mapped blocks pipeline",
            layout: sparseGridPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "clear mapped blocks module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${clearMappedBlocksSrc}`),
                }),
                entryPoint: "clearMappedBlocks",
            },
        });

        this.p2gComputePipeline = device.createComputePipeline({
            label: "particle to grid pipeline",
            layout: particlePipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "particle to grid module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${p2gModuleSrc}`),
                }),
                entryPoint: "doParticleToGrid",
            },
        });

        this.gridComputePipeline = device.createComputePipeline({
            label: "grid update pipeline",
            layout: sparseGridPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "grid update module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${gridUpdateModuleSrc}`),
                }),
                entryPoint: "doGridUpdate",
            },
        });

        this.g2pComputePipeline = device.createComputePipeline({
            label: "grid to particle pipeline",
            layout: particlePipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "grid to particle module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${g2pModuleSrc}`),
                }),
                entryPoint: "doGridToParticle",
            },
        });

        this.particleBindGroupLayout = particleBindGroupLayout;
        this.particleDataBindGroup = particleBindGroup;
        this.sparseGridBindGroupLayout = sparseGridBindGroupLayout;
        this.sparseGridBindGroup = sparseGridBindGroup;
        this.uniformsManager = uniformsManager;
    }

    addDispatch({
        computePassEncoder,
        pipeline,
        dispatchX,
        dispatchY,
        dispatchZ,
        useParticles = false,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        pipeline: GPUComputePipeline,
        dispatchX: number,
        dispatchY?: number,
        dispatchZ?: number,
        useParticles?: boolean,
    }) {
        computePassEncoder.setPipeline(pipeline);
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.sparseGridBindGroup);
        if (useParticles) {
            computePassEncoder.setBindGroup(2, this.particleDataBindGroup);
        }
        computePassEncoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    }
    
    // addIndirectDispatch({
    //     computePassEncoder,
    //     pipeline,
    //     indirectBuffer,
    //     useParticles = false,
    // }: {
    //     computePassEncoder: GPUComputePassEncoder,
    //     pipeline: GPUComputePipeline,
    //     indirectBuffer: GPUBuffer,
    //     useParticles?: boolean,
    // }) {
    //     computePassEncoder.setPipeline(pipeline);
    //     computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
    //     computePassEncoder.setBindGroup(1, this.sparseGridBindGroup);
    //     if (useParticles) {
    //         computePassEncoder.setBindGroup(2, this.particleDataBindGroup);
    //     }
    //     computePassEncoder.dispatchWorkgroupsIndirect(indirectBuffer, 0);
    // }
}