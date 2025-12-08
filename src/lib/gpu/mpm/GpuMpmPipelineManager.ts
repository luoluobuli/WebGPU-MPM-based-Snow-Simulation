import type { GpuUniformsBufferManager } from "../uniforms/GpuUniformsBufferManager";
import p2gModuleSrc from "./particleToGrid.cs.wgsl?raw";
import gridUpdateModuleSrc from "./gridUpdate.cs.wgsl?raw";
import g2pModuleSrc from "./gridToParticle.cs.wgsl?raw";
import bukkitPreludeSrc from "./bukkitPrelude.wgsl?raw";
import pbmpmSrc from "./pbmpm.wgsl?raw";
import integrateParticlesSrc from "./integrateParticles.wgsl?raw";
import bukkitCountSrc from "./bukkitCount.wgsl?raw";
import bukkitAllocateSrc from "./bukkitAllocate.wgsl?raw";
import bukkitInsertSrc from "./bukkitInsert.wgsl?raw";
import clearGridSrc from "./clearGrid.wgsl?raw";
import { attachPrelude } from "../shaderPrelude";
import type { GpuMpmBufferManager } from "./GpuMpmBufferManager";
import type { GpuColliderBufferManager } from "../collider/GpuColliderBufferManager";
import colliderPreludeModuleSrc from "./colliderPrelude.wgsl?raw";

export class GpuMpmPipelineManager {
    // Bind group layouts
    readonly uniformsBindGroupLayout: GPUBindGroupLayout;
    readonly uniformsAndParamsBindGroupLayout: GPUBindGroupLayout;
    readonly gridBindGroupLayout: GPUBindGroupLayout;
    readonly gridBindGroup: GPUBindGroup;
    readonly particleBindGroupLayout: GPUBindGroupLayout;
    readonly particleBindGroup: GPUBindGroup;
    readonly bukkitBindGroupLayout: GPUBindGroupLayout;
    readonly bukkitBindGroup: GPUBindGroup;

    // Pipelines
    readonly clearGridPipeline: GPUComputePipeline;
    readonly bukkitCountPipeline: GPUComputePipeline;
    readonly bukkitAllocatePipeline: GPUComputePipeline;
    readonly bukkitInsertPipeline: GPUComputePipeline;
    readonly p2gComputePipeline: GPUComputePipeline;
    readonly gridComputePipeline: GPUComputePipeline;
    readonly g2pComputePipeline: GPUComputePipeline;
    readonly integrateParticlesPipeline: GPUComputePipeline;
    readonly pbmpmPipeline: GPUComputePipeline;

    private readonly uniformsManager: GpuUniformsBufferManager;
    private readonly mpmManager: GpuMpmBufferManager;
    private readonly device: GPUDevice;

    constructor({
        device,
        uniformsManager,
        mpmManager,
        colliderManager,
    }: {
        device: GPUDevice,
        uniformsManager: GpuUniformsBufferManager,
        mpmManager: GpuMpmBufferManager,
        colliderManager: GpuColliderBufferManager,
    }) {
        this.device = device;
        this.uniformsManager = uniformsManager;
        this.mpmManager = mpmManager;

        uniformsManager.writeColliderNumIndices(colliderManager.numIndices);

        // Uniforms bind group layout (includes bukkit params)
        this.uniformsBindGroupLayout = device.createBindGroupLayout({
            label: "MPM uniforms bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
        });

        // 1. Uniforms + Bukkit Params Bind Group Layout
        this.uniformsAndParamsBindGroupLayout = device.createBindGroupLayout({
            label: "MPM uniforms and params bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
        });

        // Grid bind group layout (dense grid + collider)
        this.gridBindGroupLayout = device.createBindGroupLayout({
            label: "MPM grid bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // bukkitCounts / insertCounters
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // bukkitDispatch
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // bukkitThreadData
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // grid_mass
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // grid_momentum_x
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // grid_momentum_y
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // grid_momentum_z
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // bukkitParticleAllocator
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // bukkitIndexStart
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // collider
            ],
        });

        this.gridBindGroup = device.createBindGroup({
            label: "MPM grid bind group",
            layout: this.gridBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: mpmManager.bukkitCountBuffer } },
                { binding: 1, resource: { buffer: mpmManager.bukkitDispatchBuffer } },
                { binding: 2, resource: { buffer: mpmManager.bukkitThreadDataBuffer } },
                { binding: 3, resource: { buffer: mpmManager.gridMassBuffer } },
                { binding: 4, resource: { buffer: mpmManager.gridMomentumXBuffer } },
                { binding: 5, resource: { buffer: mpmManager.gridMomentumYBuffer } },
                { binding: 6, resource: { buffer: mpmManager.gridMomentumZBuffer } },
                { binding: 7, resource: { buffer: mpmManager.bukkitParticleAllocatorBuffer } },
                { binding: 8, resource: { buffer: mpmManager.bukkitIndexStartBuffer } },
                { binding: 9, resource: { buffer: colliderManager.colliderDataBuffer } },
            ],
        });

        // Bukkit-specific bind group for insert counters
        this.bukkitBindGroupLayout = device.createBindGroupLayout({
            label: "MPM bukkit bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // insertCounters
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // indexStart
            ],
        });

        this.bukkitBindGroup = device.createBindGroup({
            label: "MPM bukkit bind group",
            layout: this.bukkitBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: mpmManager.bukkitInsertCountersBuffer } },
                { binding: 4, resource: { buffer: mpmManager.bukkitIndexStartBuffer } },
            ],
        });

        // Particle bind group layout
        this.particleBindGroupLayout = device.createBindGroupLayout({
            label: "MPM particle bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });

        this.particleBindGroup = device.createBindGroup({
            label: "MPM particle bind group",
            layout: this.particleBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: mpmManager.particleDataBuffer } },
                { binding: 1, resource: { buffer: mpmManager.sortedParticleIndicesBuffer } },
            ],
        });

        // Create pipeline layouts
        const gridOnlyPipelineLayout = device.createPipelineLayout({
            label: "grid only pipeline layout",
            bindGroupLayouts: [uniformsManager.bindGroupLayout, this.gridBindGroupLayout],
        });

        const particlePipelineLayout = device.createPipelineLayout({
            label: "particle pipeline layout",
            bindGroupLayouts: [uniformsManager.bindGroupLayout, this.gridBindGroupLayout, this.particleBindGroupLayout],
        });

        const bukkitCountLayout = device.createPipelineLayout({
            label: "bukkit count pipeline layout",
            bindGroupLayouts: [
                this.uniformsAndParamsBindGroupLayout, // Group 0
                device.createBindGroupLayout({         // Group 1 (only needs bukkitCountBuffer)
                    entries: [
                        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                    ],
                }),
                this.particleBindGroupLayout,          // Group 2
            ],
        });

        const bukkitAllocateLayout = device.createPipelineLayout({
             label: "bukkit allocate pipeline layout",
             bindGroupLayouts: [
                 this.uniformsAndParamsBindGroupLayout, // Group 0
                 this.gridBindGroupLayout,             // Group 1
             ]
        });

        const bukkitInsertLayout = device.createPipelineLayout({
            label: "bukkit insert pipeline layout",
            bindGroupLayouts: [
                this.uniformsAndParamsBindGroupLayout, // Group 0
                this.bukkitBindGroupLayout,            // Group 1
                this.particleBindGroupLayout,          // Group 2
            ],
        });

        // Create shaders
        const fullPrelude = attachPrelude(bukkitPreludeSrc);

        this.clearGridPipeline = device.createComputePipeline({
            label: "clear grid pipeline",
            layout: gridOnlyPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "clear grid module",
                    code: fullPrelude + clearGridSrc,
                }),
                entryPoint: "clearGrid",
            },
        });

        this.bukkitCountPipeline = device.createComputePipeline({
            label: "bukkit count pipeline",
            layout: bukkitCountLayout,
            compute: {
                module: device.createShaderModule({
                    label: "bukkit count module",
                    code: fullPrelude + bukkitCountSrc,
                }),
                entryPoint: "bukkitCount",
            },
        });

        this.bukkitAllocatePipeline = device.createComputePipeline({
            label: "bukkit allocate pipeline",
            layout: bukkitAllocateLayout,
            compute: {
                module: device.createShaderModule({
                    label: "bukkit allocate module",
                    code: fullPrelude + bukkitAllocateSrc,
                }),
                entryPoint: "bukkitAllocate",
            },
        });

        this.bukkitInsertPipeline = device.createComputePipeline({
            label: "bukkit insert pipeline",
            layout: bukkitInsertLayout,
            compute: {
                module: device.createShaderModule({
                    label: "bukkit insert module",
                    code: fullPrelude + bukkitInsertSrc,
                }),
                entryPoint: "bukkitInsert",
            },
        });

        this.p2gComputePipeline = device.createComputePipeline({
            label: "particle to grid pipeline",
            layout: particlePipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "particle to grid module",
                    code: fullPrelude + p2gModuleSrc,
                }),
                entryPoint: "doParticleToGrid",
            },
        });

        this.gridComputePipeline = device.createComputePipeline({
            label: "grid update pipeline",
            layout: gridOnlyPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "grid update module",
                    code: fullPrelude + gridUpdateModuleSrc,
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
                    code: fullPrelude + g2pModuleSrc,
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
                    code: attachPrelude(`${colliderPreludeModuleSrc}\n${bukkitPreludeSrc}\n${integrateParticlesSrc}`),
                }),
                entryPoint: "integrateParticles",
            },
        });

        this.pbmpmPipeline = device.createComputePipeline({
            label: "pbmpm fused pipeline",
            layout: particlePipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "pbmpm fused module",
                    code: fullPrelude + pbmpmSrc,
                }),
                entryPoint: "pbmpm",
            },
        });
    }

    // Helper to add a dispatch with standard bind groups
    addDispatch({
        computePassEncoder,
        pipeline,
        dispatchX,
        dispatchY = 1,
        dispatchZ = 1,
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
        computePassEncoder.setBindGroup(1, this.gridBindGroup);
        if (useParticles) {
            computePassEncoder.setBindGroup(2, this.particleBindGroup);
        }
        computePassEncoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    }

    // PBMPM simulation step with bukkit-based spatial partitioning
    addPbmpmDispatches({
        computePassEncoder,
        nParticles,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        nParticles: number,
    }) {
        const bukkitCountX = this.mpmManager.bukkitCountX;
        const bukkitCountY = this.mpmManager.bukkitCountY;
        const bukkitCountZ = this.mpmManager.bukkitCountZ;
        const totalGridCells = this.mpmManager.totalGridCells;

        const nSolveConstraintIterations = 3;

        // Create shared bind group for uniforms + bukkitParams
        const uniformsAndParamsBindGroup = this.device.createBindGroup({
            label: "uniforms and params bind group",
            layout: this.uniformsAndParamsBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformsManager.buffer } },
                { binding: 1, resource: { buffer: this.mpmManager.bukkitParamsBuffer } },
            ],
        });

        // 1. Clear grid
        this.addDispatch({
            computePassEncoder,
            pipeline: this.clearGridPipeline,
            dispatchX: Math.ceil(totalGridCells / 256),
        });

        // Note: Clear bukkit buffers is done via encoder.clearBuffer() in the calling code

        // 2. Count particles per bukkit
        computePassEncoder.setPipeline(this.bukkitCountPipeline);
        computePassEncoder.setBindGroup(0, uniformsAndParamsBindGroup);
        computePassEncoder.setBindGroup(1, this.device.createBindGroup({
            layout: this.bukkitCountPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.mpmManager.bukkitCountBuffer } },
            ],
        }));
        computePassEncoder.setBindGroup(2, this.particleBindGroup);
        computePassEncoder.dispatchWorkgroups(Math.ceil(nParticles / 256));

        // 3. Allocate thread data
        // Uses bukkitAllocatePipeline which needs:
        // Group 0: uniformsAndParamsBindGroup
        // Group 1: gridBindGroup (includes all bukkit/grid buffers needed by allocate)
        computePassEncoder.setPipeline(this.bukkitAllocatePipeline);
        computePassEncoder.setBindGroup(0, uniformsAndParamsBindGroup);
        computePassEncoder.setBindGroup(1, this.gridBindGroup);
        computePassEncoder.dispatchWorkgroups(
            Math.ceil(bukkitCountX / 8),
            Math.ceil(bukkitCountY / 8),
            Math.ceil(bukkitCountZ / 4)
        );

        // 4. Insert particles into sorted order
        computePassEncoder.setPipeline(this.bukkitInsertPipeline);
        computePassEncoder.setBindGroup(0, uniformsAndParamsBindGroup);
        computePassEncoder.setBindGroup(1, this.bukkitBindGroup);
        computePassEncoder.setBindGroup(2, this.particleBindGroup);
        computePassEncoder.dispatchWorkgroups(Math.ceil(nParticles / 256));

        // 5. PBMPM iterations
        for (let i = 0; i < nSolveConstraintIterations; i++) {
            // Clear grid each iteration
            this.addDispatch({
                computePassEncoder,
                pipeline: this.clearGridPipeline,
                dispatchX: Math.ceil(totalGridCells / 256),
            });

            // Fused solve constraints + P2G
            this.addDispatch({
                computePassEncoder,
                pipeline: this.pbmpmPipeline,
                dispatchX: Math.ceil(nParticles / 256),
                useParticles: true,
            });

            // G2P
            this.addDispatch({
                computePassEncoder,
                pipeline: this.g2pComputePipeline,
                dispatchX: Math.ceil(nParticles / 256),
                useParticles: true,
            });
        }

        // 6. Integrate particles
        this.addDispatch({
            computePassEncoder,
            pipeline: this.integrateParticlesPipeline,
            dispatchX: Math.ceil(nParticles / 256),
            useParticles: true,
        });
    }

    // Explicit MPM (non-PBMPM) dispatches
    addExplicitMpmDispatches({
        computePassEncoder,
        nParticles,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        nParticles: number,
    }) {
        const totalGridCells = this.mpmManager.totalGridCells;

        // Clear grid
        this.addDispatch({
            computePassEncoder,
            pipeline: this.clearGridPipeline,
            dispatchX: Math.ceil(totalGridCells / 256),
        });

        // P2G
        this.addDispatch({
            computePassEncoder,
            pipeline: this.p2gComputePipeline,
            dispatchX: Math.ceil(nParticles / 256),
            useParticles: true,
        });

        // Grid update
        this.addDispatch({
            computePassEncoder,
            pipeline: this.gridComputePipeline,
            dispatchX: Math.ceil(totalGridCells / 256),
        });

        // G2P
        this.addDispatch({
            computePassEncoder,
            pipeline: this.g2pComputePipeline,
            dispatchX: Math.ceil(nParticles / 256),
            useParticles: true,
        });

        // Integrate particles
        this.addDispatch({
            computePassEncoder,
            pipeline: this.integrateParticlesPipeline,
            dispatchX: Math.ceil(nParticles / 256),
            useParticles: true,
        });
    }
}