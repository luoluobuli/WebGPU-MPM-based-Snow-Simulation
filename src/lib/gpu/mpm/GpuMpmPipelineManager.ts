import type { GpuUniformsBufferManager } from "../uniforms/GpuUniformsBufferManager";
import p2gModuleSrc from "./particleToGrid.cs.wgsl?raw";
import gridUpdateModuleSrc from "./gridUpdate.cs.wgsl?raw";
import g2pModuleSrc from "./gridToParticle.cs.wgsl?raw";
import sparseGridPreludeSrc from "./sparseGridPrelude.wgsl?raw";
import mapAffectedBlocksSrc from "./mapAffectedBlocks.wgsl?raw";
import clearBukkitSrc from "./clearBukkit.wgsl?raw";
import resetBukkitTableSrc from "./resetBukkitTable.wgsl?raw";
import clearMappedBlocksSrc from "./clearMappedBlocks.wgsl?raw";
import integrateParticlesSrc from "./integrateParticles.wgsl?raw";
import buildActiveBlockDispatchArgsSrc from "./buildActiveBlockDispatchArgs.wgsl?raw";
import { attachPrelude } from "../shaderPrelude";
import type { GpuColliderBufferManager } from "../collider/GpuColliderBufferManager";
import colliderPreludeModuleSrc from "./colliderPrelude.wgsl?raw";
import { BUKKIT_DOMAIN_BLOCK_COUNT, BUKKIT_DOMAIN_BLOCKS_PER_AXIS } from "./GpuMpmBufferManager";

const BUKKIT_GENERATION_RESERVED = 0xFFFFFFFF;

export class GpuMpmPipelineManager {
    readonly particleBindGroupLayout: GPUBindGroupLayout;
    readonly particleDataBindGroup: GPUBindGroup;
    readonly particleReadBindGroupLayout: GPUBindGroupLayout;
    readonly particleReadBindGroup: GPUBindGroup;
    readonly sparseGridBindGroupLayout: GPUBindGroupLayout;
    readonly sparseGridBindGroup: GPUBindGroup;

    readonly clearBukkitPipeline: GPUComputePipeline;
    readonly resetBukkitTablePipeline: GPUComputePipeline;
    readonly mapAffectedBlocksPipeline: GPUComputePipeline;
    readonly clearMappedBlocksPipeline: GPUComputePipeline;
    readonly explicitP2gComputePipeline: GPUComputePipeline;
    readonly mlsP2gComputePipeline: GPUComputePipeline;
    readonly gridComputePipeline: GPUComputePipeline;
    readonly interactionGridComputePipeline: GPUComputePipeline;
    readonly explicitG2pComputePipeline: GPUComputePipeline;
    readonly mlsG2pComputePipeline: GPUComputePipeline;
    readonly integrateParticlesPipeline: GPUComputePipeline;
    readonly speedRecordingIntegrateParticlesPipeline: GPUComputePipeline;
    readonly buildActiveBlockDispatchArgsPipeline: GPUComputePipeline;
    readonly activeBlockDispatchArgsBindGroup: GPUBindGroup;
    readonly activeBlockDispatchArgsReadBindGroup: GPUBindGroup;

    private readonly uniformsManager: GpuUniformsBufferManager;
    private readonly nParticleWorkgroups: number;
    private currentBukkitGeneration = 0;
    private activeBlocksPrepared = false;

    constructor({
        device,
        nParticles,
        gridResolutionX,
        gridResolutionY,
        gridResolutionZ,
        particleDataBuffer,
        sparseGridBuffer,
        gridMassBuffer,
        gridMomentumXBuffer,
        gridMomentumYBuffer,
        gridMomentumZBuffer,
        maxParticleSpeedBuffer,
        activeBlockDispatchBuffer,
        uniformsManager,
        colliderManager,
    }: {
        device: GPUDevice,
        nParticles: number,
        gridResolutionX: number,
        gridResolutionY: number,
        gridResolutionZ: number,
        particleDataBuffer: GPUBuffer,
        sparseGridBuffer: GPUBuffer,
        gridMassBuffer: GPUBuffer,
        gridMomentumXBuffer: GPUBuffer,
        gridMomentumYBuffer: GPUBuffer,
        gridMomentumZBuffer: GPUBuffer,
        maxParticleSpeedBuffer: GPUBuffer,
        activeBlockDispatchBuffer: GPUBuffer,
        uniformsManager: GpuUniformsBufferManager,
        colliderManager: GpuColliderBufferManager,
    }) {
        const particleCountConstant = nParticles;
        const nParticleWorkgroups = Math.ceil(nParticles / 256);
        const gridBoundaryMaxX = gridResolutionX - 3;
        const gridBoundaryMaxY = gridResolutionY - 3;
        const gridBoundaryMaxZ = gridResolutionZ - 3;
        const gridBoundaryHighBlockX = Math.floor((gridBoundaryMaxX + 1) / 4);
        const gridBoundaryHighBlockY = Math.floor((gridBoundaryMaxY + 1) / 4);
        const gridBoundaryHighBlockZ = Math.floor((gridBoundaryMaxZ + 1) / 4);
        const bukkitDomainMaxBlock = BUKKIT_DOMAIN_BLOCKS_PER_AXIS - 1;
        const gridLastCellX = Math.max(0, gridResolutionX - 1);
        const gridLastCellY = Math.max(0, gridResolutionY - 1);
        const gridLastCellZ = Math.max(0, gridResolutionZ - 1);
        const sparseGridConstants = {
            GRID_LAST_CELL_X: gridLastCellX,
            GRID_LAST_CELL_Y: gridLastCellY,
            GRID_LAST_CELL_Z: gridLastCellZ,
            GRID_DOMAIN_MAX_BLOCK_X: Math.min(Math.floor(gridLastCellX / 4), bukkitDomainMaxBlock),
            GRID_DOMAIN_MAX_BLOCK_Y: Math.min(Math.floor(gridLastCellY / 4), bukkitDomainMaxBlock),
            GRID_DOMAIN_MAX_BLOCK_Z: Math.min(Math.floor(gridLastCellZ / 4), bukkitDomainMaxBlock),
        };
        const sparseGridBindGroupLayout = device.createBindGroupLayout({
            label: "MPM sparse grid bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            ],
        });

        const sparseGridBindGroup = device.createBindGroup({
            label: "MPM sparse grid bind group",
            layout: sparseGridBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: sparseGridBuffer } },
                { binding: 3, resource: { buffer: gridMassBuffer } },
                { binding: 4, resource: { buffer: gridMomentumXBuffer } },
                { binding: 5, resource: { buffer: gridMomentumYBuffer } },
                { binding: 6, resource: { buffer: gridMomentumZBuffer } },
                { binding: 10, resource: { buffer: colliderManager.colliderSdfBuffer } },
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
                        buffer: maxParticleSpeedBuffer,
                    },
                },
            ],
        });

        const particleReadBindGroupLayout = device.createBindGroupLayout({
            label: "simulation step read-only particle bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage",
                    },
                },
            ],
        });

        const particleReadBindGroup = device.createBindGroup({
            label: "simulation step read-only particle bind group",
            layout: particleReadBindGroupLayout,
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

        const particleReadPipelineLayout = device.createPipelineLayout({
            label: "read-only particle pipeline layout",
            bindGroupLayouts: [uniformsManager.bindGroupLayout, sparseGridBindGroupLayout, particleReadBindGroupLayout],
        });

        const activeBlockDispatchArgsBindGroupLayout = device.createBindGroupLayout({
            label: "active block dispatch args bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });

        const activeBlockDispatchArgsBindGroup = device.createBindGroup({
            label: "active block dispatch args bind group",
            layout: activeBlockDispatchArgsBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: activeBlockDispatchBuffer } },
            ],
        });

        const activeBlockDispatchArgsReadBindGroupLayout = device.createBindGroupLayout({
            label: "active block dispatch args read bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            ],
        });

        const activeBlockDispatchArgsReadBindGroup = device.createBindGroup({
            label: "active block dispatch args read bind group",
            layout: activeBlockDispatchArgsReadBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: activeBlockDispatchBuffer } },
            ],
        });

        const activeBlockDispatchArgsPipelineLayout = device.createPipelineLayout({
            label: "active block dispatch args pipeline layout",
            bindGroupLayouts: [
                uniformsManager.bindGroupLayout,
                sparseGridBindGroupLayout,
                activeBlockDispatchArgsBindGroupLayout,
            ],
        });

        const sparseGridIndirectPipelineLayout = device.createPipelineLayout({
            label: "sparse grid indirect pipeline layout",
            bindGroupLayouts: [
                uniformsManager.bindGroupLayout,
                sparseGridBindGroupLayout,
                activeBlockDispatchArgsReadBindGroupLayout,
            ],
        });



        this.clearBukkitPipeline = device.createComputePipeline({
            label: "clear bukkit pipeline",
            layout: sparseGridPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "clear bukkit module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${clearBukkitSrc}`),
                }),
                entryPoint: "clearBukkit",
                constants: sparseGridConstants,
            },
        });

        this.mapAffectedBlocksPipeline = device.createComputePipeline({
            label: "map affected blocks pipeline",
            layout: particleReadPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "map affected blocks module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${mapAffectedBlocksSrc}`),
                }),
                entryPoint: "mapAffectedBlocks",
                constants: {
                    ...sparseGridConstants,
                    N_PARTICLES: particleCountConstant,
                },
            },
        });

        this.clearMappedBlocksPipeline = device.createComputePipeline({
            label: "clear mapped blocks pipeline",
            layout: sparseGridIndirectPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "clear mapped blocks module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${clearMappedBlocksSrc}`),
                }),
                entryPoint: "clearMappedBlocks",
                constants: sparseGridConstants,
            },
        });

        const p2gModule = device.createShaderModule({
            label: "particle to grid module",
            code: attachPrelude(`${sparseGridPreludeSrc}\n${p2gModuleSrc}`),
        });

        this.explicitP2gComputePipeline = device.createComputePipeline({
            label: "explicit particle to grid pipeline",
            layout: particleReadPipelineLayout,
            compute: {
                module: p2gModule,
                entryPoint: "doParticleToGrid",
                constants: {
                    ...sparseGridConstants,
                    N_PARTICLES: particleCountConstant,
                    USE_MLS_MPM: 0,
                    SANITIZE_PARTICLES_IN_P2G: 0,
                },
            },
        });

        this.mlsP2gComputePipeline = device.createComputePipeline({
            label: "MLS particle to grid pipeline",
            layout: particleReadPipelineLayout,
            compute: {
                module: p2gModule,
                entryPoint: "doParticleToGrid",
                constants: {
                    ...sparseGridConstants,
                    N_PARTICLES: particleCountConstant,
                    USE_MLS_MPM: 1,
                    SANITIZE_PARTICLES_IN_P2G: 0,
                },
            },
        });

        const gridUpdateModule = device.createShaderModule({
            label: "grid update module",
            code: attachPrelude(`${sparseGridPreludeSrc}\n${gridUpdateModuleSrc}`),
        });

        this.gridComputePipeline = device.createComputePipeline({
            label: "grid update pipeline",
            layout: sparseGridIndirectPipelineLayout,
            compute: {
                module: gridUpdateModule,
                entryPoint: "doGridUpdate",
                constants: {
                    ...sparseGridConstants,
                    ENABLE_INTERACTION: 0,
                    GRID_BOUNDARY_MAX_X: gridBoundaryMaxX,
                    GRID_BOUNDARY_MAX_Y: gridBoundaryMaxY,
                    GRID_BOUNDARY_MAX_Z: gridBoundaryMaxZ,
                    GRID_BOUNDARY_HIGH_BLOCK_X: gridBoundaryHighBlockX,
                    GRID_BOUNDARY_HIGH_BLOCK_Y: gridBoundaryHighBlockY,
                    GRID_BOUNDARY_HIGH_BLOCK_Z: gridBoundaryHighBlockZ,
                },
            },
        });

        this.interactionGridComputePipeline = device.createComputePipeline({
            label: "interaction grid update pipeline",
            layout: sparseGridIndirectPipelineLayout,
            compute: {
                module: gridUpdateModule,
                entryPoint: "doGridUpdate",
                constants: {
                    ...sparseGridConstants,
                    ENABLE_INTERACTION: 1,
                    GRID_BOUNDARY_MAX_X: gridBoundaryMaxX,
                    GRID_BOUNDARY_MAX_Y: gridBoundaryMaxY,
                    GRID_BOUNDARY_MAX_Z: gridBoundaryMaxZ,
                    GRID_BOUNDARY_HIGH_BLOCK_X: gridBoundaryHighBlockX,
                    GRID_BOUNDARY_HIGH_BLOCK_Y: gridBoundaryHighBlockY,
                    GRID_BOUNDARY_HIGH_BLOCK_Z: gridBoundaryHighBlockZ,
                },
            },
        });

        const g2pModule = device.createShaderModule({
            label: "grid to particle module",
            code: attachPrelude(`${sparseGridPreludeSrc}\n${g2pModuleSrc}`),
        });

        this.explicitG2pComputePipeline = device.createComputePipeline({
            label: "explicit grid to particle pipeline",
            layout: particlePipelineLayout,
            compute: {
                module: g2pModule,
                entryPoint: "doGridToParticle",
                constants: {
                    ...sparseGridConstants,
                    N_PARTICLES: particleCountConstant,
                    USE_MLS_MPM: 0,
                },
            },
        });

        this.mlsG2pComputePipeline = device.createComputePipeline({
            label: "MLS grid to particle pipeline",
            layout: particlePipelineLayout,
            compute: {
                module: g2pModule,
                entryPoint: "doGridToParticle",
                constants: {
                    ...sparseGridConstants,
                    N_PARTICLES: particleCountConstant,
                    USE_MLS_MPM: 1,
                },
            },
        });

        const integrateParticlesModule = device.createShaderModule({
            label: "integrate particles module",
            code: attachPrelude(`${colliderPreludeModuleSrc}\n${sparseGridPreludeSrc}\n${integrateParticlesSrc}`),
        });

        this.integrateParticlesPipeline = device.createComputePipeline({
            label: "integrate particles pipeline",
            layout: particlePipelineLayout,
            compute: {
                module: integrateParticlesModule,
                entryPoint: "integrateParticles",
                constants: {
                    ...sparseGridConstants,
                    N_PARTICLES: particleCountConstant,
                    RECORD_PARTICLE_SPEED: 0,
                },
            },
        });

        this.speedRecordingIntegrateParticlesPipeline = device.createComputePipeline({
            label: "speed recording integrate particles pipeline",
            layout: particlePipelineLayout,
            compute: {
                module: integrateParticlesModule,
                entryPoint: "integrateParticles",
                constants: {
                    ...sparseGridConstants,
                    N_PARTICLES: particleCountConstant,
                    RECORD_PARTICLE_SPEED: 1,
                },
            },
        });

        this.resetBukkitTablePipeline = device.createComputePipeline({
            label: "reset bukkit table pipeline",
            layout: sparseGridPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "reset bukkit table module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${resetBukkitTableSrc}`),
                }),
                entryPoint: "resetBukkitTable",
                constants: sparseGridConstants,
            },
        });

        this.buildActiveBlockDispatchArgsPipeline = device.createComputePipeline({
            label: "build active block dispatch args pipeline",
            layout: activeBlockDispatchArgsPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "build active block dispatch args module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${buildActiveBlockDispatchArgsSrc}`),
                }),
                entryPoint: "buildActiveBlockDispatchArgs",
                constants: sparseGridConstants,
            },
        });

        this.particleBindGroupLayout = particleBindGroupLayout;
        this.particleDataBindGroup = particleBindGroup;
        this.particleReadBindGroupLayout = particleReadBindGroupLayout;
        this.particleReadBindGroup = particleReadBindGroup;
        this.sparseGridBindGroupLayout = sparseGridBindGroupLayout;
        this.sparseGridBindGroup = sparseGridBindGroup;
        this.activeBlockDispatchArgsBindGroup = activeBlockDispatchArgsBindGroup;
        this.activeBlockDispatchArgsReadBindGroup = activeBlockDispatchArgsReadBindGroup;
        this.uniformsManager = uniformsManager;
        this.nParticleWorkgroups = nParticleWorkgroups;
    }

    addDispatch({
        computePassEncoder,
        pipeline,
        dispatchX,
        dispatchY,
        dispatchZ,
        useParticles = false,
        bindParticleGroup = useParticles,
        bindCommonGroups = true,
        particleBindGroup,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        pipeline: GPUComputePipeline,
        dispatchX: number,
        dispatchY?: number,
        dispatchZ?: number,
        useParticles?: boolean,
        bindParticleGroup?: boolean,
        bindCommonGroups?: boolean,
        particleBindGroup?: GPUBindGroup,
    }) {
        computePassEncoder.setPipeline(pipeline);
        if (bindCommonGroups) {
            this.bindCommonComputeGroups(computePassEncoder);
        }
        if (bindParticleGroup) {
            computePassEncoder.setBindGroup(2, particleBindGroup ?? this.particleDataBindGroup);
        }
        computePassEncoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    }

    addActiveBlockDispatchArgsDispatch({
        computePassEncoder,
        bindCommonGroups = true,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        bindCommonGroups?: boolean,
    }) {
        computePassEncoder.setPipeline(this.buildActiveBlockDispatchArgsPipeline);
        if (bindCommonGroups) {
            this.bindCommonComputeGroups(computePassEncoder);
        }
        computePassEncoder.setBindGroup(2, this.activeBlockDispatchArgsBindGroup);
        computePassEncoder.dispatchWorkgroups(1);
    }

    addIndirectDispatch({
        computePassEncoder,
        pipeline,
        indirectBuffer,
        indirectOffset = 0,
        bindActiveBlockDispatchArgs = true,
        bindCommonGroups = true,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        pipeline: GPUComputePipeline,
        indirectBuffer: GPUBuffer,
        indirectOffset?: number,
        bindActiveBlockDispatchArgs?: boolean,
        bindCommonGroups?: boolean,
    }) {
        computePassEncoder.setPipeline(pipeline);
        if (bindCommonGroups) {
            this.bindCommonComputeGroups(computePassEncoder);
        }
        if (bindActiveBlockDispatchArgs) {
            computePassEncoder.setBindGroup(2, this.activeBlockDispatchArgsReadBindGroup);
        }
        computePassEncoder.dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset);
    }

    bindCommonComputeGroups(computePassEncoder: GPUComputePassEncoder) {
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.sparseGridBindGroup);
    }

    invalidateActiveBlocks() {
        this.activeBlocksPrepared = false;
    }

    addBukkitResetDispatch({
        computePassEncoder,
        bindCommonGroups = true,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        bindCommonGroups?: boolean,
    }) {
        if (this.currentBukkitGeneration >= BUKKIT_GENERATION_RESERVED - 1) {
            this.currentBukkitGeneration = 1;
            computePassEncoder.setPipeline(this.resetBukkitTablePipeline);
            if (bindCommonGroups) {
                this.bindCommonComputeGroups(computePassEncoder);
            }
            computePassEncoder.dispatchWorkgroups(Math.ceil(BUKKIT_DOMAIN_BLOCK_COUNT / 256));
            return;
        }

        this.currentBukkitGeneration += 1;
        computePassEncoder.setPipeline(this.clearBukkitPipeline);
        if (bindCommonGroups) {
            this.bindCommonComputeGroups(computePassEncoder);
        }
        computePassEncoder.dispatchWorkgroups(1);
    }

    private encodeBukkitResetDispatch(computePassEncoder: GPUComputePassEncoder) {
        if (this.currentBukkitGeneration >= BUKKIT_GENERATION_RESERVED - 1) {
            this.currentBukkitGeneration = 1;
            computePassEncoder.setPipeline(this.resetBukkitTablePipeline);
            computePassEncoder.dispatchWorkgroups(Math.ceil(BUKKIT_DOMAIN_BLOCK_COUNT / 256));
            return;
        }

        this.currentBukkitGeneration += 1;
        computePassEncoder.setPipeline(this.clearBukkitPipeline);
        computePassEncoder.dispatchWorkgroups(1);
    }

    addExplicitMpmDispatches({
        computePassEncoder,
        activeBlockDispatchBuffer,
        enableInteraction,
        recordParticleSpeed,
        bindCommonGroups = true,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        activeBlockDispatchBuffer: GPUBuffer,
        enableInteraction: boolean,
        recordParticleSpeed: boolean,
        bindCommonGroups?: boolean,
    }) {
        this.addMpmDispatches({
            computePassEncoder,
            activeBlockDispatchBuffer,
            gridPipeline: enableInteraction ? this.interactionGridComputePipeline : this.gridComputePipeline,
            p2gPipeline: this.explicitP2gComputePipeline,
            g2pPipeline: this.explicitG2pComputePipeline,
            integratePipeline: recordParticleSpeed
                ? this.speedRecordingIntegrateParticlesPipeline
                : this.integrateParticlesPipeline,
            bindCommonGroups,
        });
    }

    addExplicitMpmDispatchesBatch({
        computePassEncoder,
        activeBlockDispatchBuffer,
        enableInteraction,
        nSimulationSteps,
        recordParticleSpeed,
        bindCommonGroups = true,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        activeBlockDispatchBuffer: GPUBuffer,
        enableInteraction: boolean,
        nSimulationSteps: number,
        recordParticleSpeed: boolean,
        bindCommonGroups?: boolean,
    }) {
        this.addMpmDispatchesBatch({
            computePassEncoder,
            activeBlockDispatchBuffer,
            gridPipeline: enableInteraction ? this.interactionGridComputePipeline : this.gridComputePipeline,
            p2gPipeline: this.explicitP2gComputePipeline,
            g2pPipeline: this.explicitG2pComputePipeline,
            normalIntegratePipeline: this.integrateParticlesPipeline,
            speedRecordingIntegratePipeline: this.speedRecordingIntegrateParticlesPipeline,
            nSimulationSteps,
            recordParticleSpeed,
            bindCommonGroups,
        });
    }

    addMpmDispatches({
        computePassEncoder,
        activeBlockDispatchBuffer,
        gridPipeline,
        p2gPipeline,
        g2pPipeline,
        integratePipeline,
        bindCommonGroups = true,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        activeBlockDispatchBuffer: GPUBuffer,
        gridPipeline: GPUComputePipeline,
        p2gPipeline: GPUComputePipeline,
        g2pPipeline: GPUComputePipeline,
        integratePipeline: GPUComputePipeline,
        bindCommonGroups?: boolean,
    }) {
        if (bindCommonGroups) {
            this.bindCommonComputeGroups(computePassEncoder);
        }

        this.encodeMpmDispatches(
            computePassEncoder,
            activeBlockDispatchBuffer,
            gridPipeline,
            p2gPipeline,
            g2pPipeline,
            integratePipeline,
        );
    }

    private addMpmDispatchesBatch({
        computePassEncoder,
        activeBlockDispatchBuffer,
        gridPipeline,
        p2gPipeline,
        g2pPipeline,
        normalIntegratePipeline,
        speedRecordingIntegratePipeline,
        nSimulationSteps,
        recordParticleSpeed,
        bindCommonGroups = true,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        activeBlockDispatchBuffer: GPUBuffer,
        gridPipeline: GPUComputePipeline,
        p2gPipeline: GPUComputePipeline,
        g2pPipeline: GPUComputePipeline,
        normalIntegratePipeline: GPUComputePipeline,
        speedRecordingIntegratePipeline: GPUComputePipeline,
        nSimulationSteps: number,
        recordParticleSpeed: boolean,
        bindCommonGroups?: boolean,
    }) {
        if (nSimulationSteps <= 0) return;

        if (bindCommonGroups) {
            this.bindCommonComputeGroups(computePassEncoder);
        }

        const normalStepCount = recordParticleSpeed
            ? nSimulationSteps - 1
            : nSimulationSteps;

        for (let i = 0; i < normalStepCount; i++) {
            this.encodeMpmDispatches(
                computePassEncoder,
                activeBlockDispatchBuffer,
                gridPipeline,
                p2gPipeline,
                g2pPipeline,
                normalIntegratePipeline,
            );
        }

        if (recordParticleSpeed) {
            this.encodeMpmDispatches(
                computePassEncoder,
                activeBlockDispatchBuffer,
                gridPipeline,
                p2gPipeline,
                g2pPipeline,
                speedRecordingIntegratePipeline,
            );
        }
    }

    private encodeMpmDispatches(
        computePassEncoder: GPUComputePassEncoder,
        activeBlockDispatchBuffer: GPUBuffer,
        gridPipeline: GPUComputePipeline,
        p2gPipeline: GPUComputePipeline,
        g2pPipeline: GPUComputePipeline,
        integratePipeline: GPUComputePipeline,
    ) {
        if (!this.activeBlocksPrepared) {
            this.encodeBukkitResetDispatch(computePassEncoder);

            computePassEncoder.setPipeline(this.mapAffectedBlocksPipeline);
            computePassEncoder.setBindGroup(2, this.particleReadBindGroup);
            computePassEncoder.dispatchWorkgroups(this.nParticleWorkgroups);
        }

        computePassEncoder.setPipeline(this.buildActiveBlockDispatchArgsPipeline);
        computePassEncoder.setBindGroup(2, this.activeBlockDispatchArgsBindGroup);
        computePassEncoder.dispatchWorkgroups(1);

        // clear cells
        computePassEncoder.setPipeline(this.clearMappedBlocksPipeline);
        computePassEncoder.setBindGroup(2, this.activeBlockDispatchArgsReadBindGroup);
        computePassEncoder.dispatchWorkgroupsIndirect(activeBlockDispatchBuffer, 16);

        
        // particle-to-grid

        computePassEncoder.setPipeline(p2gPipeline);
        computePassEncoder.setBindGroup(2, this.particleReadBindGroup);
        computePassEncoder.dispatchWorkgroups(this.nParticleWorkgroups);

        // grid update

        computePassEncoder.setPipeline(gridPipeline);
        computePassEncoder.setBindGroup(2, this.activeBlockDispatchArgsReadBindGroup);
        computePassEncoder.dispatchWorkgroupsIndirect(activeBlockDispatchBuffer, 0);

        // grid-to-particle

        computePassEncoder.setPipeline(g2pPipeline);
        computePassEncoder.setBindGroup(2, this.particleDataBindGroup);
        computePassEncoder.dispatchWorkgroups(this.nParticleWorkgroups);

        this.encodeBukkitResetDispatch(computePassEncoder);

        // G2P left group 2 bound to particle data, and the bukkit reset above
        // only uses groups 0-1. Reuse it here to avoid a bind command per substep.
        computePassEncoder.setPipeline(integratePipeline);
        computePassEncoder.dispatchWorkgroups(this.nParticleWorkgroups);
        this.activeBlocksPrepared = true;
    }


    addMlsMpmDispatches({
        computePassEncoder,
        activeBlockDispatchBuffer,
        enableInteraction,
        recordParticleSpeed,
        bindCommonGroups = true,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        activeBlockDispatchBuffer: GPUBuffer,
        enableInteraction: boolean,
        recordParticleSpeed: boolean,
        bindCommonGroups?: boolean,
    }) {
        this.addMpmDispatches({
            computePassEncoder,
            activeBlockDispatchBuffer,
            gridPipeline: enableInteraction ? this.interactionGridComputePipeline : this.gridComputePipeline,
            p2gPipeline: this.mlsP2gComputePipeline,
            g2pPipeline: this.mlsG2pComputePipeline,
            integratePipeline: recordParticleSpeed
                ? this.speedRecordingIntegrateParticlesPipeline
                : this.integrateParticlesPipeline,
            bindCommonGroups,
        });
    }

    addMlsMpmDispatchesBatch({
        computePassEncoder,
        activeBlockDispatchBuffer,
        enableInteraction,
        nSimulationSteps,
        recordParticleSpeed,
        bindCommonGroups = true,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        activeBlockDispatchBuffer: GPUBuffer,
        enableInteraction: boolean,
        nSimulationSteps: number,
        recordParticleSpeed: boolean,
        bindCommonGroups?: boolean,
    }) {
        this.addMpmDispatchesBatch({
            computePassEncoder,
            activeBlockDispatchBuffer,
            gridPipeline: enableInteraction ? this.interactionGridComputePipeline : this.gridComputePipeline,
            p2gPipeline: this.mlsP2gComputePipeline,
            g2pPipeline: this.mlsG2pComputePipeline,
            normalIntegratePipeline: this.integrateParticlesPipeline,
            speedRecordingIntegratePipeline: this.speedRecordingIntegrateParticlesPipeline,
            nSimulationSteps,
            recordParticleSpeed,
            bindCommonGroups,
        });
    }
}
