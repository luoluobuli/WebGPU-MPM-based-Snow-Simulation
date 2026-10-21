import type { GpuUniformsBufferManager } from "../uniforms/GpuUniformsBufferManager";
import pbmpmOpsSrc from "./pbmpmOps.wgsl?raw";
import { normalizePbmpmSolveIterations } from "./pbmpmSettings";
import p2gModuleSrc from "./particleToGrid.cs.wgsl?raw";
import gridUpdateModuleSrc from "./gridUpdate.cs.wgsl?raw";
import g2pModuleSrc from "./gridToParticle.cs.wgsl?raw";
import sparseGridPreludeSrc from "./sparseGridPrelude.wgsl?raw";
import mapAffectedBlocksSrc from "./mapAffectedBlocks.wgsl?raw";
import clearBukkitSrc from "./clearBukkit.wgsl?raw";
import clearGridAccumulatorSrc from "./clearGridAccumulator.wgsl?raw";
import resetBukkitTableSrc from "./resetBukkitTable.wgsl?raw";
import integrateParticlesSrc from "./integrateParticles.wgsl?raw";
import { attachPrelude } from "../shaderPrelude";
import type { GpuColliderBufferManager } from "../collider/GpuColliderBufferManager";
import colliderPreludeModuleSrc from "./colliderPrelude.wgsl?raw";
import { BUKKIT_DOMAIN_BLOCK_COUNT, BUKKIT_DOMAIN_BLOCKS_PER_AXIS } from "./GpuMpmBufferManager";
import type {
    GpuRigidPawSdfKinematicColliderManager,
} from "../kinematicCollider/GpuRigidPawSdfKinematicColliderManager";
import rigidPawSdfColliderPreludeSrc from "../kinematicCollider/rigidPawSdfKinematicColliderPrelude.wgsl?raw";

const BUKKIT_GENERATION_RESERVED = 0xFFFFFFFF;
const PARTICLE_WORKGROUP_SIZE = 256;

export class GpuMpmPipelineManager {
    readonly particleBindGroupLayout: GPUBindGroupLayout;
    readonly particleDataBindGroup: GPUBindGroup;
    readonly particleReadBindGroupLayout: GPUBindGroupLayout;
    readonly particleReadBindGroup: GPUBindGroup;
    readonly sparseGridBindGroupLayout: GPUBindGroupLayout;
    readonly sparseGridBindGroup: GPUBindGroup;

    readonly clearBukkitPipeline: GPUComputePipeline;
    readonly clearGridAccumulatorPipeline: GPUComputePipeline;
    readonly resetBukkitTablePipeline: GPUComputePipeline;
    readonly mapAffectedBlocksPipeline: GPUComputePipeline;
    readonly explicitP2gComputePipeline: GPUComputePipeline;
    readonly mlsP2gComputePipeline: GPUComputePipeline;
    readonly pbmpmP2gComputePipeline: GPUComputePipeline;
    readonly pbmpmIterationGridComputePipeline: GPUComputePipeline;
    readonly pbmpmGridComputePipeline: GPUComputePipeline;
    readonly pbmpmInteractionGridComputePipeline: GPUComputePipeline;
    readonly gridComputePipeline: GPUComputePipeline;
    readonly interactionGridComputePipeline: GPUComputePipeline;
    readonly explicitG2pComputePipeline: GPUComputePipeline;
    readonly mlsG2pComputePipeline: GPUComputePipeline;
    readonly integrateParticlesPipeline: GPUComputePipeline;
    readonly speedRecordingIntegrateParticlesPipeline: GPUComputePipeline;
    readonly validColliderIntegrateParticlesPipeline: GPUComputePipeline;
    readonly validColliderSpeedRecordingIntegrateParticlesPipeline: GPUComputePipeline;
    readonly staticColliderIntegrateParticlesPipeline: GPUComputePipeline;
    readonly staticColliderSpeedRecordingIntegrateParticlesPipeline: GPUComputePipeline;
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
        particleFlagsBuffer,
        sparseGridBuffer,
        gridAccumulatorBuffer,
        gridVelocityBuffer,
        maxParticleSpeedBuffer,
        activeBlockDispatchBuffer,
        uniformsManager,
        colliderManager,
        kinematicColliderManager,
    }: {
        device: GPUDevice,
        nParticles: number,
        gridResolutionX: number,
        gridResolutionY: number,
        gridResolutionZ: number,
        particleDataBuffer: GPUBuffer,
        particleFlagsBuffer: GPUBuffer,
        sparseGridBuffer: GPUBuffer,
        gridAccumulatorBuffer: GPUBuffer,
        gridVelocityBuffer: GPUBuffer,
        maxParticleSpeedBuffer: GPUBuffer,
        activeBlockDispatchBuffer: GPUBuffer,
        uniformsManager: GpuUniformsBufferManager,
        colliderManager: GpuColliderBufferManager,
        kinematicColliderManager: GpuRigidPawSdfKinematicColliderManager,
    }) {
        const particleCountConstant = nParticles;
        const nParticleWorkgroups = Math.ceil(nParticles / PARTICLE_WORKGROUP_SIZE);
        const particleKernelConstants = {
            N_PARTICLES: particleCountConstant,
            PARTICLE_WORKGROUP_SIZE,
        };
        const kinematicColliderConstants = {
            KINEMATIC_COLLIDER_COUNT:
                kinematicColliderManager.primitiveCount,
            KINEMATIC_COLLIDER_SDF_RESOLUTION:
                kinematicColliderManager.sdfResolution,
        };
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
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            ],
        });

        const createSparseGridBindGroup = (
            label: string,
            sparseBuffer: GPUBuffer,
            accumulatorBuffer: GPUBuffer,
            velocityBuffer: GPUBuffer,
        ) => device.createBindGroup({
            label,
            layout: sparseGridBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: sparseBuffer } },
                { binding: 3, resource: { buffer: accumulatorBuffer } },
                { binding: 7, resource: { buffer: velocityBuffer } },
                { binding: 10, resource: { buffer: colliderManager.colliderSdfBuffer } },
                { binding: 11, resource: { buffer: kinematicColliderManager.poseBuffer } },
                { binding: 12, resource: { buffer: kinematicColliderManager.sdfBuffer } },
            ],
        });
        const sparseGridBindGroup = createSparseGridBindGroup(
            "MPM sparse grid bind group",
            sparseGridBuffer,
            gridAccumulatorBuffer,
            gridVelocityBuffer,
        );
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
                {
                    binding: 2,
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
                {
                    binding: 2,
                    resource: {
                        buffer: particleFlagsBuffer,
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
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage",
                    },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
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
                {
                    binding: 2,
                    resource: {
                        buffer: particleFlagsBuffer,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: activeBlockDispatchBuffer,
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

        const activeBlockDispatchArgsReadBindGroupLayout = device.createBindGroupLayout({
            label: "active block dispatch args read bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            ],
        });

        const createActiveBlockDispatchArgsReadBindGroup = (
            label: string,
            activeBlockDispatchArgsBuffer: GPUBuffer,
        ) => device.createBindGroup({
            label,
            layout: activeBlockDispatchArgsReadBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: activeBlockDispatchArgsBuffer } },
            ],
        });
        const activeBlockDispatchArgsReadBindGroup = createActiveBlockDispatchArgsReadBindGroup(
            "active block dispatch args read bind group",
            activeBlockDispatchBuffer,
        );

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

        this.clearGridAccumulatorPipeline = device.createComputePipeline({
            label: "clear grid accumulator pipeline",
            layout: sparseGridPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "clear grid accumulator module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${clearGridAccumulatorSrc}`),
                }),
                entryPoint: "clearGridAccumulator",
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
                    ...particleKernelConstants,
                },
            },
        });

        const p2gModule = device.createShaderModule({
            label: "particle to grid module",
            code: attachPrelude(`${sparseGridPreludeSrc}\n${pbmpmOpsSrc}\n${p2gModuleSrc}`),
        });

        this.explicitP2gComputePipeline = device.createComputePipeline({
            label: "explicit particle to grid pipeline",
            layout: particleReadPipelineLayout,
            compute: {
                module: p2gModule,
                entryPoint: "doParticleToGrid",
                constants: {
                    ...sparseGridConstants,
                    ...particleKernelConstants,
                    USE_MLS_MPM: 0,
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
                    ...particleKernelConstants,
                    USE_MLS_MPM: 1,
                },
            },
        });

        this.pbmpmP2gComputePipeline = device.createComputePipeline({
            label: "PBMPM particle to grid pipeline",
            layout: particleReadPipelineLayout,
            compute: {
                module: p2gModule,
                entryPoint: "doParticleToGrid",
                constants: {
                    ...sparseGridConstants,
                    ...particleKernelConstants,
                    USE_MLS_MPM: 1,
                    USE_PBMPM: 1,
                },
            },
        });

        const gridUpdateModule = device.createShaderModule({
            label: "grid update module",
            code: attachPrelude(
                `${colliderPreludeModuleSrc}\n${rigidPawSdfColliderPreludeSrc}\n${sparseGridPreludeSrc}\n${gridUpdateModuleSrc}`,
            ),
        });

        const createGridPipeline = (
            label: string,
            enableInteraction: boolean,
            applyExternalForces: boolean,
            usePbmpmContact: boolean,
        ) => device.createComputePipeline({
            label,
            layout: sparseGridIndirectPipelineLayout,
            compute: {
                module: gridUpdateModule,
                entryPoint: "doGridUpdate",
                constants: {
                    ...sparseGridConstants,
                    ...kinematicColliderConstants,
                    ENABLE_INTERACTION: enableInteraction ? 1 : 0,
                    APPLY_EXTERNAL_FORCES: applyExternalForces ? 1 : 0,
                    USE_PBMPM_CONTACT: usePbmpmContact ? 1 : 0,
                    GRID_BOUNDARY_MAX_X: gridBoundaryMaxX,
                    GRID_BOUNDARY_MAX_Y: gridBoundaryMaxY,
                    GRID_BOUNDARY_MAX_Z: gridBoundaryMaxZ,
                    GRID_BOUNDARY_HIGH_BLOCK_X: gridBoundaryHighBlockX,
                    GRID_BOUNDARY_HIGH_BLOCK_Y: gridBoundaryHighBlockY,
                    GRID_BOUNDARY_HIGH_BLOCK_Z: gridBoundaryHighBlockZ,
                },
            },
        });
        this.gridComputePipeline = createGridPipeline("grid update pipeline", false, true, false);
        this.interactionGridComputePipeline = createGridPipeline("interaction grid update pipeline", true, true, false);
        this.pbmpmGridComputePipeline = createGridPipeline("PBMPM grid update pipeline", false, true, true);
        this.pbmpmInteractionGridComputePipeline = createGridPipeline("PBMPM interaction grid update pipeline", true, true, true);
        this.pbmpmIterationGridComputePipeline = createGridPipeline("PBMPM iteration grid update pipeline", false, false, true);

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
                    ...particleKernelConstants,
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
                    ...particleKernelConstants,
                    USE_MLS_MPM: 1,
                },
            },
        });

        const integrateParticlesModule = device.createShaderModule({
            label: "integrate particles module",
            code: attachPrelude(
                `${colliderPreludeModuleSrc}\n${rigidPawSdfColliderPreludeSrc}\n${sparseGridPreludeSrc}\n${integrateParticlesSrc}`,
            ),
        });

        const createIntegrateParticlesPipeline = ({
            label,
            recordParticleSpeed,
            particleSpeedReductionWorkgroupSize,
            colliderTransformAlwaysIdentity,
            colliderVelocityAlwaysZero,
            colliderSdfAlwaysValid,
        }: {
            label: string,
            recordParticleSpeed: boolean,
            particleSpeedReductionWorkgroupSize: number,
            colliderTransformAlwaysIdentity: boolean,
            colliderVelocityAlwaysZero: boolean,
            colliderSdfAlwaysValid: boolean,
        }) => device.createComputePipeline({
            label,
            layout: particlePipelineLayout,
            compute: {
                module: integrateParticlesModule,
                entryPoint: "integrateParticles",
                constants: {
                    ...sparseGridConstants,
                    ...particleKernelConstants,
                    ...kinematicColliderConstants,
                    RECORD_PARTICLE_SPEED: recordParticleSpeed ? 1 : 0,
                    PARTICLE_SPEED_REDUCTION_WORKGROUP_SIZE: particleSpeedReductionWorkgroupSize,
                    COLLIDER_TRANSFORM_ALWAYS_IDENTITY: colliderTransformAlwaysIdentity ? 1 : 0,
                    COLLIDER_VELOCITY_ALWAYS_ZERO: colliderVelocityAlwaysZero ? 1 : 0,
                    COLLIDER_SDF_ALWAYS_VALID: colliderSdfAlwaysValid ? 1 : 0,
                },
            },
        });

        this.integrateParticlesPipeline = createIntegrateParticlesPipeline({
            label: "integrate particles pipeline",
            recordParticleSpeed: false,
            particleSpeedReductionWorkgroupSize: 1,
            colliderTransformAlwaysIdentity: false,
            colliderVelocityAlwaysZero: false,
            colliderSdfAlwaysValid: false,
        });

        this.speedRecordingIntegrateParticlesPipeline = createIntegrateParticlesPipeline({
            label: "speed recording integrate particles pipeline",
            recordParticleSpeed: true,
            particleSpeedReductionWorkgroupSize: PARTICLE_WORKGROUP_SIZE,
            colliderTransformAlwaysIdentity: false,
            colliderVelocityAlwaysZero: false,
            colliderSdfAlwaysValid: false,
        });

        this.validColliderIntegrateParticlesPipeline = createIntegrateParticlesPipeline({
            label: "valid collider integrate particles pipeline",
            recordParticleSpeed: false,
            particleSpeedReductionWorkgroupSize: 1,
            colliderTransformAlwaysIdentity: false,
            colliderVelocityAlwaysZero: false,
            colliderSdfAlwaysValid: true,
        });

        this.validColliderSpeedRecordingIntegrateParticlesPipeline = createIntegrateParticlesPipeline({
            label: "valid collider speed recording integrate particles pipeline",
            recordParticleSpeed: true,
            particleSpeedReductionWorkgroupSize: PARTICLE_WORKGROUP_SIZE,
            colliderTransformAlwaysIdentity: false,
            colliderVelocityAlwaysZero: false,
            colliderSdfAlwaysValid: true,
        });

        this.staticColliderIntegrateParticlesPipeline = createIntegrateParticlesPipeline({
            label: "static collider integrate particles pipeline",
            recordParticleSpeed: false,
            particleSpeedReductionWorkgroupSize: 1,
            colliderTransformAlwaysIdentity: true,
            colliderVelocityAlwaysZero: true,
            colliderSdfAlwaysValid: true,
        });

        this.staticColliderSpeedRecordingIntegrateParticlesPipeline = createIntegrateParticlesPipeline({
            label: "static collider speed recording integrate particles pipeline",
            recordParticleSpeed: true,
            particleSpeedReductionWorkgroupSize: PARTICLE_WORKGROUP_SIZE,
            colliderTransformAlwaysIdentity: true,
            colliderVelocityAlwaysZero: true,
            colliderSdfAlwaysValid: true,
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

        this.particleBindGroupLayout = particleBindGroupLayout;
        this.particleDataBindGroup = particleBindGroup;
        this.particleReadBindGroupLayout = particleReadBindGroupLayout;
        this.particleReadBindGroup = particleReadBindGroup;
        this.sparseGridBindGroupLayout = sparseGridBindGroupLayout;
        this.sparseGridBindGroup = sparseGridBindGroup;
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
        useStaticColliderPipeline = false,
        useValidColliderPipeline = false,
        bindCommonGroups = true,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        activeBlockDispatchBuffer: GPUBuffer,
        enableInteraction: boolean,
        recordParticleSpeed: boolean,
        useStaticColliderPipeline?: boolean,
        useValidColliderPipeline?: boolean,
        bindCommonGroups?: boolean,
    }) {
        this.addMpmDispatches({
            computePassEncoder,
            activeBlockDispatchBuffer,
            gridPipeline: enableInteraction ? this.interactionGridComputePipeline : this.gridComputePipeline,
            p2gPipeline: this.explicitP2gComputePipeline,
            g2pPipeline: this.explicitG2pComputePipeline,
            integratePipeline: this.selectIntegratePipeline(
                recordParticleSpeed,
                useStaticColliderPipeline,
                useValidColliderPipeline,
            ),
            bindCommonGroups,
        });
    }

    addExplicitMpmDispatchesBatch({
        computePassEncoder,
        activeBlockDispatchBuffer,
        enableInteraction,
        nSimulationSteps,
        recordParticleSpeed,
        useStaticColliderPipeline = false,
        useValidColliderPipeline = false,
        bindCommonGroups = true,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        activeBlockDispatchBuffer: GPUBuffer,
        enableInteraction: boolean,
        nSimulationSteps: number,
        recordParticleSpeed: boolean,
        useStaticColliderPipeline?: boolean,
        useValidColliderPipeline?: boolean,
        bindCommonGroups?: boolean,
    }) {
        this.addMpmDispatchesBatch({
            computePassEncoder,
            activeBlockDispatchBuffer,
            gridPipeline: enableInteraction ? this.interactionGridComputePipeline : this.gridComputePipeline,
            p2gPipeline: this.explicitP2gComputePipeline,
            g2pPipeline: this.explicitG2pComputePipeline,
            normalIntegratePipeline: this.selectIntegratePipeline(
                false,
                useStaticColliderPipeline,
                useValidColliderPipeline,
            ),
            speedRecordingIntegratePipeline: this.selectIntegratePipeline(
                true,
                useStaticColliderPipeline,
                useValidColliderPipeline,
            ),
            nSimulationSteps,
            recordParticleSpeed,
            bindCommonGroups,
        });
    }

    private selectIntegratePipeline(
        recordParticleSpeed: boolean,
        useStaticColliderPipeline: boolean,
        useValidColliderPipeline: boolean,
    ) {
        if (recordParticleSpeed) {
            return useStaticColliderPipeline
                ? this.staticColliderSpeedRecordingIntegrateParticlesPipeline
                : useValidColliderPipeline
                    ? this.validColliderSpeedRecordingIntegrateParticlesPipeline
                    : this.speedRecordingIntegrateParticlesPipeline;
        }

        return useStaticColliderPipeline
            ? this.staticColliderIntegrateParticlesPipeline
            : useValidColliderPipeline
                ? this.validColliderIntegrateParticlesPipeline
                : this.integrateParticlesPipeline;
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

    private encodeGridAccumulatorClearDispatch(
        computePassEncoder: GPUComputePassEncoder,
        activeBlockDispatchBuffer: GPUBuffer,
    ) {
        this.bindCommonComputeGroups(computePassEncoder);
        computePassEncoder.setPipeline(this.clearGridAccumulatorPipeline);
        computePassEncoder.dispatchWorkgroupsIndirect(activeBlockDispatchBuffer, 0);
    }

    private encodeMpmDispatches(
        computePassEncoder: GPUComputePassEncoder,
        activeBlockDispatchBuffer: GPUBuffer,
        gridPipeline: GPUComputePipeline,
        p2gPipeline: GPUComputePipeline,
        g2pPipeline: GPUComputePipeline,
        integratePipeline: GPUComputePipeline,
        solveIterations = 1,
    ) {
        let particleReadBindGroupBound = false;
        if (!this.activeBlocksPrepared) {
            // clear cells
            this.encodeGridAccumulatorClearDispatch(computePassEncoder, activeBlockDispatchBuffer);
            // clear bukkit table
            this.encodeBukkitResetDispatch(computePassEncoder);

            // determine which blocks in a grid are populated
            computePassEncoder.setPipeline(this.mapAffectedBlocksPipeline);
            computePassEncoder.setBindGroup(2, this.particleReadBindGroup);
            particleReadBindGroupBound = true;
            computePassEncoder.dispatchWorkgroups(this.nParticleWorkgroups);
        }

        for (let iteration = 0; iteration < solveIterations; iteration++) {
            // Grid update consumes and clears the accumulator after every scatter.
            computePassEncoder.setPipeline(p2gPipeline);
            if (!particleReadBindGroupBound) {
                computePassEncoder.setBindGroup(2, this.particleReadBindGroup);
            }
            computePassEncoder.dispatchWorkgroups(this.nParticleWorkgroups);

            // grid update

            computePassEncoder.setPipeline(iteration === 0 ? gridPipeline : this.pbmpmIterationGridComputePipeline);
            computePassEncoder.setBindGroup(2, this.activeBlockDispatchArgsReadBindGroup);
            computePassEncoder.dispatchWorkgroupsIndirect(activeBlockDispatchBuffer, 0);

            // grid-to-particle (needs global synchronization)

            computePassEncoder.setPipeline(g2pPipeline);
            computePassEncoder.setBindGroup(2, this.particleDataBindGroup);
            computePassEncoder.dispatchWorkgroups(this.nParticleWorkgroups);
            particleReadBindGroupBound = false;
        }

        this.encodeBukkitResetDispatch(computePassEncoder);

        // Integrate Particles (Update Pos + Deformation + Handle Collision)
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
        useStaticColliderPipeline = false,
        useValidColliderPipeline = false,
        bindCommonGroups = true,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        activeBlockDispatchBuffer: GPUBuffer,
        enableInteraction: boolean,
        recordParticleSpeed: boolean,
        useStaticColliderPipeline?: boolean,
        useValidColliderPipeline?: boolean,
        bindCommonGroups?: boolean,
    }) {
        this.addMpmDispatches({
            computePassEncoder,
            activeBlockDispatchBuffer,
            gridPipeline: enableInteraction ? this.interactionGridComputePipeline : this.gridComputePipeline,
            p2gPipeline: this.mlsP2gComputePipeline,
            g2pPipeline: this.mlsG2pComputePipeline,
            integratePipeline: this.selectIntegratePipeline(
                recordParticleSpeed,
                useStaticColliderPipeline,
                useValidColliderPipeline,
            ),
            bindCommonGroups,
        });
    }

    addMlsMpmDispatchesBatch({
        computePassEncoder,
        activeBlockDispatchBuffer,
        enableInteraction,
        nSimulationSteps,
        recordParticleSpeed,
        useStaticColliderPipeline = false,
        useValidColliderPipeline = false,
        bindCommonGroups = true,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        activeBlockDispatchBuffer: GPUBuffer,
        enableInteraction: boolean,
        nSimulationSteps: number,
        recordParticleSpeed: boolean,
        useStaticColliderPipeline?: boolean,
        useValidColliderPipeline?: boolean,
        bindCommonGroups?: boolean,
    }) {
        this.addMpmDispatchesBatch({
            computePassEncoder,
            activeBlockDispatchBuffer,
            gridPipeline: enableInteraction ? this.interactionGridComputePipeline : this.gridComputePipeline,
            p2gPipeline: this.mlsP2gComputePipeline,
            g2pPipeline: this.mlsG2pComputePipeline,
            normalIntegratePipeline: this.selectIntegratePipeline(
                false,
                useStaticColliderPipeline,
                useValidColliderPipeline,
            ),
            speedRecordingIntegratePipeline: this.selectIntegratePipeline(
                true,
                useStaticColliderPipeline,
                useValidColliderPipeline,
            ),
            nSimulationSteps,
            recordParticleSpeed,
            bindCommonGroups,
        });
    }

    addPbmpmDispatchesBatch({
        computePassEncoder,
        activeBlockDispatchBuffer,
        enableInteraction,
        nSimulationSteps,
        solveIterations,
        recordParticleSpeed,
        useStaticColliderPipeline = false,
        useValidColliderPipeline = false,
        bindCommonGroups = true,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        activeBlockDispatchBuffer: GPUBuffer,
        enableInteraction: boolean,
        nSimulationSteps: number,
        solveIterations: number,
        recordParticleSpeed: boolean,
        useStaticColliderPipeline?: boolean,
        useValidColliderPipeline?: boolean,
        bindCommonGroups?: boolean,
    }) {
        if (nSimulationSteps <= 0) return;
        if (bindCommonGroups) this.bindCommonComputeGroups(computePassEncoder);

        const iterations = normalizePbmpmSolveIterations(solveIterations);
        for (let step = 0; step < nSimulationSteps; step++) {
            this.encodeMpmDispatches(
                computePassEncoder,
                activeBlockDispatchBuffer,
                enableInteraction ? this.pbmpmInteractionGridComputePipeline : this.pbmpmGridComputePipeline,
                this.pbmpmP2gComputePipeline,
                this.mlsG2pComputePipeline,
                this.selectIntegratePipeline(
                    recordParticleSpeed && step === nSimulationSteps - 1,
                    useStaticColliderPipeline,
                    useValidColliderPipeline,
                ),
                iterations,
            );
        }
    }
}
