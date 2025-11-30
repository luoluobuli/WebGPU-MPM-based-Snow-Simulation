import type { GpuUniformsBufferManager } from "../uniforms/GpuUniformsBufferManager";
import p2gModuleSrc from "./particleToGrid.cs.wgsl?raw";
import gridUpdateModuleSrc from "./gridUpdate.cs.wgsl?raw";
import g2pModuleSrc from "./gridToParticle.cs.wgsl?raw";
import sparseGridPreludeSrc from "./sparseGridPrelude.wgsl?raw";
import mapAffectedBlocksSrc from "./mapAffectedBlocks.wgsl?raw";
import clearHashMapSrc from "./clearHashMap.wgsl?raw";
import clearMappedBlocksSrc from "./clearMappedBlocks.wgsl?raw";
import solveParticleConstraintsSrc from "./solveParticleConstraints.wgsl?raw";
import integrateParticlesSrc from "./integrateParticles.wgsl?raw";
import { attachPrelude } from "../shaderPrelude";

export class GpuMpmPipelineManager {
    readonly particleBindGroupLayout: GPUBindGroupLayout;
    readonly particleDataBindGroup: GPUBindGroup;
    readonly sparseGridBindGroupLayout: GPUBindGroupLayout;
    readonly sparseGridBindGroup: GPUBindGroup;

    readonly solveParticleConstraintsPipeline: GPUComputePipeline;
    readonly clearHashMapPipeline: GPUComputePipeline;
    readonly mapAffectedBlocksPipeline: GPUComputePipeline;
    readonly clearMappedBlocksPipeline: GPUComputePipeline;
    readonly p2gComputePipeline: GPUComputePipeline;
    readonly gridComputePipeline: GPUComputePipeline;
    readonly g2pComputePipeline: GPUComputePipeline;
    readonly integrateParticlesPipeline: GPUComputePipeline;

    private readonly uniformsManager: GpuUniformsBufferManager;

    constructor({
        device,
        particleDataBuffer,
        pageTableBuffer,
        gridMassBuffer,
        gridMomentumXBuffer,
        gridMomentumYBuffer,
        gridMomentumZBuffer,
        gridDisplacementXBuffer,
        gridDisplacementYBuffer,
        gridDisplacementZBuffer,
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
        gridDisplacementXBuffer: GPUBuffer,
        gridDisplacementYBuffer: GPUBuffer,
        gridDisplacementZBuffer: GPUBuffer,
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
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
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
                { binding: 7, resource: { buffer: gridDisplacementXBuffer } },
                { binding: 8, resource: { buffer: gridDisplacementYBuffer } },
                { binding: 9, resource: { buffer: gridDisplacementZBuffer } },
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

        this.solveParticleConstraintsPipeline = device.createComputePipeline({
            label: "solve particle constraints pipeline",
            layout: particlePipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "solve particle constraints module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${solveParticleConstraintsSrc}`),
                }),
                entryPoint: "solveParticleConstraints",
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

        this.integrateParticlesPipeline = device.createComputePipeline({
            label: "integrate particles pipeline",
            layout: particlePipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "integrate particles module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${integrateParticlesSrc}`),
                }),
                entryPoint: "integrateParticles",
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

    addExplicitMpmDispatches({
        computePassEncoder,
        hashMapSize,
        nBlocksInHashMap,
        nParticles,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        hashMapSize: number,
        nBlocksInHashMap: number,
        nParticles: number,
    }) {
        const gridCellDispatchX = 256;
        const gridCellDispatchY = Math.ceil(nBlocksInHashMap / gridCellDispatchX);

        // clear grid

        // clear mapping table
        this.addDispatch({
            computePassEncoder,
            pipeline: this.clearHashMapPipeline,
            dispatchX: Math.ceil(hashMapSize / 256),
        });

        // determine which blocks in a grid are populated
        this.addDispatch({
            computePassEncoder,
            pipeline: this.mapAffectedBlocksPipeline,
            dispatchX: Math.ceil(nParticles / 256),
            useParticles: true,
        });

        // clear cells
        this.addDispatch({
            computePassEncoder,
            pipeline: this.clearMappedBlocksPipeline,
            dispatchX: gridCellDispatchX,
            dispatchY: gridCellDispatchY,
        });

        
        // particle-to-grid

        this.addDispatch({
            computePassEncoder,
            pipeline: this.p2gComputePipeline,
            dispatchX: Math.ceil(nParticles / 256),
            useParticles: true,
        });

        // grid update

        this.addDispatch({
            computePassEncoder,
            pipeline: this.gridComputePipeline,
            dispatchX: gridCellDispatchX,
            dispatchY: gridCellDispatchY,
        });

        // grid-to-particle

        this.addDispatch({
            computePassEncoder,
            pipeline: this.g2pComputePipeline,
            dispatchX: Math.ceil(nParticles / 256),
            useParticles: true,
        });
    }


    addPbmpmDispatches({
        computePassEncoder,
        nParticles,
        nBlocksInHashMap,
        hashMapSize,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        nParticles: number,
        nBlocksInHashMap: number,
        hashMapSize: number,
    }) {
        const gridCellDispatchX = 256;
        const gridCellDispatchY = Math.ceil(nBlocksInHashMap / gridCellDispatchX);

        const nSolveConstraintIterations = 3;

        for (let i = 0; i < nSolveConstraintIterations; i++) {
            // solve constraints
            this.addDispatch({
                computePassEncoder,
                pipeline: this.solveParticleConstraintsPipeline,
                dispatchX: Math.ceil(nParticles / 256),
                useParticles: true,
            });

            // clear grid

            // clear mapping table
            this.addDispatch({
                computePassEncoder,
                pipeline: this.clearHashMapPipeline,
                dispatchX: Math.ceil(hashMapSize / 256),
            });

            // determine which blocks in a grid are populated
            this.addDispatch({
                computePassEncoder,
                pipeline: this.mapAffectedBlocksPipeline,
                dispatchX: Math.ceil(nParticles / 256),
                useParticles: true,
            });

            // clear cells
            this.addDispatch({
                computePassEncoder,
                pipeline: this.clearMappedBlocksPipeline,
                dispatchX: gridCellDispatchX,
                dispatchY: gridCellDispatchY,
            });
        
            // particle-to-grid
            this.addDispatch({
                computePassEncoder,
                pipeline: this.p2gComputePipeline,
                dispatchX: Math.ceil(nParticles / 256),
                useParticles: true,
            });

            // grid update
            this.addDispatch({
                computePassEncoder,
                pipeline: this.gridComputePipeline,
                dispatchX: gridCellDispatchX,
                dispatchY: gridCellDispatchY,
            });

            // grid-to-particle
            this.addDispatch({
                computePassEncoder,
                pipeline: this.g2pComputePipeline,
                dispatchX: Math.ceil(nParticles / 256),
                useParticles: true,
            });
        }

        this.addDispatch({
            computePassEncoder,
            pipeline: this.integrateParticlesPipeline,
            dispatchX: Math.ceil(nParticles / 256),
            useParticles: true,
        });
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