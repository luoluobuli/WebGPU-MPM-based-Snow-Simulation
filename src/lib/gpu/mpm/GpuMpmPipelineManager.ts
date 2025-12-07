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
import countParticlesPerBlockSrc from "./countParticlesPerBlock.wgsl?raw";
import computeBlockOffsetsSrc from "./computeBlockOffsets.wgsl?raw";
import binParticlesSrc from "./binParticles.wgsl?raw";
import clearBlockParticleCountsSrc from "./clearBlockParticleCounts.wgsl?raw";
import { attachPrelude } from "../shaderPrelude";
import type { GpuMpmBufferManager } from "./GpuMpmBufferManager";
import type { GpuColliderBufferManager } from "../collider/GpuColliderBufferManager";
import colliderPreludeModuleSrc from "./colliderPrelude.wgsl?raw";

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
    readonly countParticlesPerBlockPipeline: GPUComputePipeline;
    readonly computeBlockOffsetsPipeline: GPUComputePipeline;
    readonly binParticlesPipeline: GPUComputePipeline;
    readonly clearBlockParticleCountsPipeline: GPUComputePipeline;

    private readonly uniformsManager: GpuUniformsBufferManager;
    private readonly mpmManager: GpuMpmBufferManager;

    constructor({
        device,
        particleDataBuffer,
        sparseGridBuffer,
        gridMassBuffer,
        gridMomentumXBuffer,
        gridMomentumYBuffer,
        gridMomentumZBuffer,
        sortedParticleIndicesBuffer,
        uniformsManager,
        mpmManager,
        colliderManager,
    }: {
        device: GPUDevice,
        particleDataBuffer: GPUBuffer,
        sparseGridBuffer: GPUBuffer,
        gridMassBuffer: GPUBuffer,
        gridMomentumXBuffer: GPUBuffer,
        gridMomentumYBuffer: GPUBuffer,
        gridMomentumZBuffer: GPUBuffer,
        sortedParticleIndicesBuffer: GPUBuffer,
        uniformsManager: GpuUniformsBufferManager,
        mpmManager: GpuMpmBufferManager,
        colliderManager: GpuColliderBufferManager,
    }) {
        const sparseGridBindGroupLayout = device.createBindGroupLayout({
            label: "MPM sparse grid bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            ],
        });

        uniformsManager.writeColliderNumIndices(colliderManager.numIndices);


        const sparseGridBindGroup = device.createBindGroup({
            label: "MPM sparse grid bind group",
            layout: sparseGridBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: sparseGridBuffer } },
                { binding: 3, resource: { buffer: gridMassBuffer } },
                { binding: 4, resource: { buffer: gridMomentumXBuffer } },
                { binding: 5, resource: { buffer: gridMomentumYBuffer } },
                { binding: 6, resource: { buffer: gridMomentumZBuffer } },
                { binding: 9, resource: { buffer: colliderManager.colliderDataBuffer } },
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
                {
                    binding: 1,
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
                {
                    binding: 1,
                    resource: {
                        buffer: sortedParticleIndicesBuffer,
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
                    code: attachPrelude(`${colliderPreludeModuleSrc}\n${sparseGridPreludeSrc}\n${integrateParticlesSrc}`),
                }),
                entryPoint: "integrateParticles",
            },
        });

        this.countParticlesPerBlockPipeline = device.createComputePipeline({
            label: "count particles per block pipeline",
            layout: particlePipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "count particles per block module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${countParticlesPerBlockSrc}`),
                }),
                entryPoint: "countParticlesPerBlock",
            },
        });

        this.computeBlockOffsetsPipeline = device.createComputePipeline({
            label: "compute block offsets pipeline",
            layout: sparseGridPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "compute block offsets module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${computeBlockOffsetsSrc}`),
                }),
                entryPoint: "computeBlockOffsets",
            },
        });

        this.binParticlesPipeline = device.createComputePipeline({
            label: "bin particles pipeline",
            layout: particlePipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "bin particles module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${binParticlesSrc}`),
                }),
                entryPoint: "binParticles",
            },
        });

        this.clearBlockParticleCountsPipeline = device.createComputePipeline({
            label: "clear block particle counts pipeline",
            layout: sparseGridPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "clear block particle counts module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${clearBlockParticleCountsSrc}`),
                }),
                entryPoint: "clearBlockParticleCounts",
            },
        });

        this.particleBindGroupLayout = particleBindGroupLayout;
        this.particleDataBindGroup = particleBindGroup;
        this.sparseGridBindGroupLayout = sparseGridBindGroupLayout;
        this.sparseGridBindGroup = sparseGridBindGroup;
        this.uniformsManager = uniformsManager;
        this.mpmManager = mpmManager;
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

        // sort particles
        this.addDispatch({
            computePassEncoder,
            pipeline: this.clearBlockParticleCountsPipeline,
            dispatchX: Math.ceil(this.mpmManager.nMaxBlocksInHashMap / 256),
        });

        this.addDispatch({
            computePassEncoder,
            pipeline: this.countParticlesPerBlockPipeline,
            dispatchX: Math.ceil(nParticles / 256),
            useParticles: true,
        });

        this.addDispatch({
            computePassEncoder,
            pipeline: this.computeBlockOffsetsPipeline,
            dispatchX: 1,
        });

        this.addDispatch({
            computePassEncoder,
            pipeline: this.binParticlesPipeline,
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

        // Integrate Particles (Update Pos + Deformation + Handle Collision)
        this.addDispatch({
            computePassEncoder,
            pipeline: this.integrateParticlesPipeline,
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

        // determine which blocks in a grid are populated

        // clear mapping table
        this.addDispatch({
            computePassEncoder,
            pipeline: this.clearHashMapPipeline,
            dispatchX: Math.ceil(hashMapSize / 256),
        });

        this.addDispatch({
            computePassEncoder,
            pipeline: this.mapAffectedBlocksPipeline,
            dispatchX: Math.ceil(nParticles / 256),
            useParticles: true,
        });

        // Sort particles
        this.addDispatch({
            computePassEncoder,
            pipeline: this.clearBlockParticleCountsPipeline,
            dispatchX: Math.ceil(100000 / 256), // nMaxBlocksInHashMap
        });

        this.addDispatch({
            computePassEncoder,
            pipeline: this.countParticlesPerBlockPipeline,
            dispatchX: Math.ceil(nParticles / 256),
            useParticles: true,
        });

        this.addDispatch({
            computePassEncoder,
            pipeline: this.computeBlockOffsetsPipeline,
            dispatchX: 1,
        });

        this.addDispatch({
            computePassEncoder,
            pipeline: this.binParticlesPipeline,
            dispatchX: Math.ceil(nParticles / 256),
            useParticles: true,
        });

        for (let i = 0; i < nSolveConstraintIterations; i++) {
            // solve constraints
            this.addDispatch({
                computePassEncoder,
                pipeline: this.solveParticleConstraintsPipeline,
                dispatchX: Math.ceil(nParticles / 256),
                useParticles: true,
            });

            // clear grid
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