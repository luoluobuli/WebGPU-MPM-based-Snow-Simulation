import type { GpuUniformsBufferManager } from "../uniforms/GpuUniformsBufferManager";
import p2gModuleSrc from "./particleToGrid.cs.wgsl?raw";
import gridUpdateModuleSrc from "./gridUpdate.cs.wgsl?raw";
import g2pModuleSrc from "./gridToParticle.cs.wgsl?raw";
import sparseGridPreludeSrc from "./sparseGridPrelude.wgsl?raw";
import mapAffectedBlocksSrc from "./mapAffectedBlocks.wgsl?raw";
import clearBukkitSrc from "./clearBukkit.wgsl?raw";
import resetBukkitTableSrc from "./resetBukkitTable.wgsl?raw";
import integrateParticlesSrc from "./integrateParticles.wgsl?raw";
import bukkitBuildSrc from "./bukkitBuild.wgsl?raw";
import fusedMlsG2p2gSrc from "./fusedMlsG2p2g.cs.wgsl?raw";
import finalizeNextActiveBlocksSrc from "./finalizeNextActiveBlocks.wgsl?raw";
import { attachPrelude } from "../shaderPrelude";
import type { GpuColliderBufferManager } from "../collider/GpuColliderBufferManager";
import colliderPreludeModuleSrc from "./colliderPrelude.wgsl?raw";
import { BUKKIT_DOMAIN_BLOCK_COUNT, BUKKIT_DOMAIN_BLOCKS_PER_AXIS, N_MAX_ACTIVE_BLOCKS } from "./GpuMpmBufferManager";

const BUKKIT_GENERATION_RESERVED = 0xFFFFFFFF;
const PARTICLE_WORKGROUP_SIZE = 256;

export class GpuMpmPipelineManager {
    readonly particleBindGroupLayout: GPUBindGroupLayout;
    readonly particleDataBindGroup: GPUBindGroup;
    readonly particleReadBindGroupLayout: GPUBindGroupLayout;
    readonly particleReadBindGroup: GPUBindGroup;
    readonly sparseGridBindGroupLayout: GPUBindGroupLayout;
    readonly sparseGridBindGroup: GPUBindGroup;
    readonly nextSparseGridBindGroup: GPUBindGroup;

    readonly clearBukkitPipeline: GPUComputePipeline;
    readonly resetBukkitTablePipeline: GPUComputePipeline;
    readonly resetBukkitBuildBuffersPipeline: GPUComputePipeline;
    readonly countParticlesPerBukkitPipeline: GPUComputePipeline;
    readonly allocateBukkitThreadDataPipeline: GPUComputePipeline;
    readonly insertParticlesIntoBukkitPipeline: GPUComputePipeline;
    readonly finalizeBukkitDispatchPipeline: GPUComputePipeline;
    readonly mapAffectedBlocksPipeline: GPUComputePipeline;
    readonly explicitP2gComputePipeline: GPUComputePipeline;
    readonly mlsP2gComputePipeline: GPUComputePipeline;
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
    readonly fusedMlsG2p2gPipeline: GPUComputePipeline;
    readonly fusedMlsSpeedRecordingG2p2gPipeline: GPUComputePipeline;
    readonly finalizeNextActiveBlockDispatchPipeline: GPUComputePipeline;
    readonly activeBlockDispatchArgsReadBindGroup: GPUBindGroup;
    readonly nextActiveBlockDispatchArgsReadBindGroup: GPUBindGroup;

    private readonly uniformsManager: GpuUniformsBufferManager;
    private readonly nParticleWorkgroups: number;
    private readonly nBukkitWorkgroups: number;
    private readonly nActiveBlockWorkgroups: number;
    private readonly fusedSparseGridBindGroups: GPUBindGroup[];
    private readonly fusedActiveBlockDispatchArgsReadBindGroups: GPUBindGroup[];
    private readonly fusedBukkitBindGroups: GPUBindGroup[];
    private readonly fusedG2p2gBindGroups: GPUBindGroup[];
    private readonly fusedActiveBlockDispatchBuffers: GPUBuffer[];
    private readonly bukkitDispatchBuffer: GPUBuffer;
    private readonly bukkitParticleReadBindGroups: GPUBindGroup[];
    private currentBukkitGeneration = 0;
    private fusedBukkitGenerations = [0, 0];
    private activeBlocksPrepared = false;
    private fusedActiveBlocksPrepared = false;
    private fusedSourceSparseGridIndex = 0;
    private dispatchMode: "classic" | "fused" | null = null;

    constructor({
        device,
        nParticles,
        gridResolutionX,
        gridResolutionY,
        gridResolutionZ,
        particleDataBuffer,
        particleFlagsBuffer,
        sparseGridBuffer,
        nextSparseGridBuffer,
        gridAccumulatorBuffer,
        gridVelocityBuffer,
        maxParticleSpeedBuffer,
        activeBlockDispatchBuffer,
        nextActiveBlockDispatchBuffer,
        bukkitParticleCountsBuffer,
        bukkitInsertCountersBuffer,
        bukkitIndexStartBuffer,
        bukkitThreadDataBuffer,
        bukkitParticleDataBuffer,
        bukkitDispatchBuffer,
        bukkitParticleAllocatorBuffer,
        bukkitThreadGroupCountBuffer,
        uniformsManager,
        colliderManager,
    }: {
        device: GPUDevice,
        nParticles: number,
        gridResolutionX: number,
        gridResolutionY: number,
        gridResolutionZ: number,
        particleDataBuffer: GPUBuffer,
        particleFlagsBuffer: GPUBuffer,
        sparseGridBuffer: GPUBuffer,
        nextSparseGridBuffer: GPUBuffer,
        gridAccumulatorBuffer: GPUBuffer,
        gridVelocityBuffer: GPUBuffer,
        maxParticleSpeedBuffer: GPUBuffer,
        activeBlockDispatchBuffer: GPUBuffer,
        nextActiveBlockDispatchBuffer: GPUBuffer,
        bukkitParticleCountsBuffer: GPUBuffer,
        bukkitInsertCountersBuffer: GPUBuffer,
        bukkitIndexStartBuffer: GPUBuffer,
        bukkitThreadDataBuffer: GPUBuffer,
        bukkitParticleDataBuffer: GPUBuffer,
        bukkitDispatchBuffer: GPUBuffer,
        bukkitParticleAllocatorBuffer: GPUBuffer,
        bukkitThreadGroupCountBuffer: GPUBuffer,
        uniformsManager: GpuUniformsBufferManager,
        colliderManager: GpuColliderBufferManager,
    }) {
        const particleCountConstant = nParticles;
        const nParticleWorkgroups = Math.ceil(nParticles / PARTICLE_WORKGROUP_SIZE);
        const nBukkitWorkgroups = Math.ceil(BUKKIT_DOMAIN_BLOCK_COUNT / 256);
        const nActiveBlockWorkgroups = Math.ceil(N_MAX_ACTIVE_BLOCKS / 256);
        const particleKernelConstants = {
            N_PARTICLES: particleCountConstant,
            PARTICLE_WORKGROUP_SIZE,
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
            ],
        });

        const createSparseGridBindGroup = (label: string, sparseBuffer: GPUBuffer) => device.createBindGroup({
            label,
            layout: sparseGridBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: sparseBuffer } },
                { binding: 3, resource: { buffer: gridAccumulatorBuffer } },
                { binding: 7, resource: { buffer: gridVelocityBuffer } },
                { binding: 10, resource: { buffer: colliderManager.colliderSdfBuffer } },
            ],
        });
        const sparseGridBindGroup = createSparseGridBindGroup(
            "MPM sparse grid bind group",
            sparseGridBuffer,
        );
        const nextSparseGridBindGroup = createSparseGridBindGroup(
            "MPM fused next sparse grid bind group",
            nextSparseGridBuffer,
        );
        const fusedSparseGridBindGroups = [sparseGridBindGroup, nextSparseGridBindGroup];



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

        const bukkitParticleReadBindGroupLayout = device.createBindGroupLayout({
            label: "bukkit particle read bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage",
                    },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage",
                    },
                },
            ],
        });

        const createBukkitParticleReadBindGroup = (
            label: string,
            activeBlockDispatchArgsBuffer: GPUBuffer,
        ) => device.createBindGroup({
            label,
            layout: bukkitParticleReadBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: particleDataBuffer,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: activeBlockDispatchArgsBuffer,
                    },
                },
            ],
        });
        const bukkitParticleReadBindGroups = [
            createBukkitParticleReadBindGroup(
                "bukkit particle read bind group for primary active blocks",
                activeBlockDispatchBuffer,
            ),
            createBukkitParticleReadBindGroup(
                "bukkit particle read bind group for fused next active blocks",
                nextActiveBlockDispatchBuffer,
            ),
        ];


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
        const nextActiveBlockDispatchArgsReadBindGroup = createActiveBlockDispatchArgsReadBindGroup(
            "fused next active block dispatch args read bind group",
            nextActiveBlockDispatchBuffer,
        );
        const fusedActiveBlockDispatchArgsReadBindGroups = [
            activeBlockDispatchArgsReadBindGroup,
            nextActiveBlockDispatchArgsReadBindGroup,
        ];
        const fusedActiveBlockDispatchBuffers = [
            activeBlockDispatchBuffer,
            nextActiveBlockDispatchBuffer,
        ];

        const sparseGridIndirectPipelineLayout = device.createPipelineLayout({
            label: "sparse grid indirect pipeline layout",
            bindGroupLayouts: [
                uniformsManager.bindGroupLayout,
                sparseGridBindGroupLayout,
                activeBlockDispatchArgsReadBindGroupLayout,
            ],
        });

        const fusedBukkitBindGroupLayout = device.createBindGroupLayout({
            label: "MPM fused bukkit bind group layout",
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

        const createFusedBukkitBindGroup = (
            label: string,
            nextSparseBuffer: GPUBuffer,
            nextActiveBlockDispatchArgsBuffer: GPUBuffer,
        ) => device.createBindGroup({
            label,
            layout: fusedBukkitBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: nextSparseBuffer } },
                { binding: 1, resource: { buffer: nextActiveBlockDispatchArgsBuffer } },
                { binding: 2, resource: { buffer: bukkitParticleCountsBuffer } },
                { binding: 3, resource: { buffer: bukkitInsertCountersBuffer } },
                { binding: 4, resource: { buffer: bukkitIndexStartBuffer } },
                { binding: 5, resource: { buffer: bukkitThreadDataBuffer } },
                { binding: 6, resource: { buffer: bukkitParticleDataBuffer } },
                { binding: 7, resource: { buffer: bukkitDispatchBuffer } },
                { binding: 8, resource: { buffer: bukkitParticleAllocatorBuffer } },
                { binding: 9, resource: { buffer: bukkitThreadGroupCountBuffer } },
            ],
        });
        const fusedBukkitBindGroups = [
            createFusedBukkitBindGroup(
                "MPM fused bukkit bind group writing primary sparse grid",
                sparseGridBuffer,
                activeBlockDispatchBuffer,
            ),
            createFusedBukkitBindGroup(
                "MPM fused bukkit bind group writing next sparse grid",
                nextSparseGridBuffer,
                nextActiveBlockDispatchBuffer,
            ),
        ];

        const fusedG2p2gBindGroupLayout = device.createBindGroupLayout({
            label: "MPM fused G2P2G bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });

        const createFusedG2p2gBindGroup = (
            label: string,
            nextSparseBuffer: GPUBuffer,
        ) => device.createBindGroup({
            label,
            layout: fusedG2p2gBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: nextSparseBuffer } },
                { binding: 5, resource: { buffer: bukkitThreadDataBuffer } },
                { binding: 6, resource: { buffer: bukkitParticleDataBuffer } },
                { binding: 9, resource: { buffer: bukkitThreadGroupCountBuffer } },
            ],
        });
        const fusedG2p2gBindGroups = [
            createFusedG2p2gBindGroup(
                "MPM fused G2P2G bind group writing primary sparse grid",
                sparseGridBuffer,
            ),
            createFusedG2p2gBindGroup(
                "MPM fused G2P2G bind group writing next sparse grid",
                nextSparseGridBuffer,
            ),
        ];

        const fusedPipelineLayout = device.createPipelineLayout({
            label: "fused MLS MPM pipeline layout",
            bindGroupLayouts: [
                uniformsManager.bindGroupLayout,
                sparseGridBindGroupLayout,
                particleBindGroupLayout,
                fusedG2p2gBindGroupLayout,
            ],
        });

        const bukkitBuildPipelineLayout = device.createPipelineLayout({
            label: "bukkit build pipeline layout",
            bindGroupLayouts: [
                uniformsManager.bindGroupLayout,
                sparseGridBindGroupLayout,
                bukkitParticleReadBindGroupLayout,
                fusedBukkitBindGroupLayout,
            ],
        });

        const finalizeNextActiveBlockDispatchPipelineLayout = device.createPipelineLayout({
            label: "finalize fused next active block dispatch pipeline layout",
            bindGroupLayouts: [
                uniformsManager.bindGroupLayout,
                sparseGridBindGroupLayout,
                bukkitParticleReadBindGroupLayout,
                fusedBukkitBindGroupLayout,
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
                    ...particleKernelConstants,
                },
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
                    ...particleKernelConstants,
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
                    ...particleKernelConstants,
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
            code: attachPrelude(`${colliderPreludeModuleSrc}\n${sparseGridPreludeSrc}\n${integrateParticlesSrc}`),
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

        const bukkitBuildModule = device.createShaderModule({
            label: "bukkit build module",
            code: attachPrelude(`${sparseGridPreludeSrc}\n${bukkitBuildSrc}`),
        });
        const bukkitBuildConstants = {
            ...sparseGridConstants,
            ...particleKernelConstants,
        };

        this.resetBukkitBuildBuffersPipeline = device.createComputePipeline({
            label: "reset bukkit build buffers pipeline",
            layout: bukkitBuildPipelineLayout,
            compute: {
                module: bukkitBuildModule,
                entryPoint: "resetBukkitBuildBuffers",
                constants: bukkitBuildConstants,
            },
        });

        this.countParticlesPerBukkitPipeline = device.createComputePipeline({
            label: "count particles per bukkit pipeline",
            layout: bukkitBuildPipelineLayout,
            compute: {
                module: bukkitBuildModule,
                entryPoint: "countParticlesPerBukkit",
                constants: bukkitBuildConstants,
            },
        });

        this.allocateBukkitThreadDataPipeline = device.createComputePipeline({
            label: "allocate bukkit thread data pipeline",
            layout: bukkitBuildPipelineLayout,
            compute: {
                module: bukkitBuildModule,
                entryPoint: "allocateBukkitThreadData",
                constants: bukkitBuildConstants,
            },
        });

        this.insertParticlesIntoBukkitPipeline = device.createComputePipeline({
            label: "insert particles into bukkit pipeline",
            layout: bukkitBuildPipelineLayout,
            compute: {
                module: bukkitBuildModule,
                entryPoint: "insertParticlesIntoBukkit",
                constants: bukkitBuildConstants,
            },
        });

        this.finalizeBukkitDispatchPipeline = device.createComputePipeline({
            label: "finalize bukkit dispatch pipeline",
            layout: bukkitBuildPipelineLayout,
            compute: {
                module: bukkitBuildModule,
                entryPoint: "finalizeBukkitDispatch",
                constants: bukkitBuildConstants,
            },
        });

        const fusedMlsG2p2gModule = device.createShaderModule({
            label: "fused MLS G2P2G module",
            code: attachPrelude(`${colliderPreludeModuleSrc}\n${sparseGridPreludeSrc}\n${fusedMlsG2p2gSrc}`),
        });

        const createFusedMlsG2p2gPipeline = ({
            label,
            recordParticleSpeed,
        }: {
            label: string,
            recordParticleSpeed: boolean,
        }) => device.createComputePipeline({
            label,
            layout: fusedPipelineLayout,
            compute: {
                module: fusedMlsG2p2gModule,
                entryPoint: "doFusedMlsG2p2g",
                constants: {
                    ...sparseGridConstants,
                    ...particleKernelConstants,
                    RECORD_PARTICLE_SPEED: recordParticleSpeed ? 1 : 0,
                    PARTICLE_SPEED_REDUCTION_WORKGROUP_SIZE: PARTICLE_WORKGROUP_SIZE,
                },
            },
        });

        this.fusedMlsG2p2gPipeline = createFusedMlsG2p2gPipeline({
            label: "fused MLS G2P2G pipeline",
            recordParticleSpeed: false,
        });
        this.fusedMlsSpeedRecordingG2p2gPipeline = createFusedMlsG2p2gPipeline({
            label: "fused MLS speed recording G2P2G pipeline",
            recordParticleSpeed: true,
        });

        this.finalizeNextActiveBlockDispatchPipeline = device.createComputePipeline({
            label: "finalize fused next active block dispatch pipeline",
            layout: finalizeNextActiveBlockDispatchPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    label: "finalize fused next active block dispatch module",
                    code: attachPrelude(`${sparseGridPreludeSrc}\n${finalizeNextActiveBlocksSrc}`),
                }),
                entryPoint: "finalizeNextActiveBlockDispatch",
                constants: sparseGridConstants,
            },
        });

        this.particleBindGroupLayout = particleBindGroupLayout;
        this.particleDataBindGroup = particleBindGroup;
        this.particleReadBindGroupLayout = particleReadBindGroupLayout;
        this.particleReadBindGroup = particleReadBindGroup;
        this.sparseGridBindGroupLayout = sparseGridBindGroupLayout;
        this.sparseGridBindGroup = sparseGridBindGroup;
        this.nextSparseGridBindGroup = nextSparseGridBindGroup;
        this.activeBlockDispatchArgsReadBindGroup = activeBlockDispatchArgsReadBindGroup;
        this.nextActiveBlockDispatchArgsReadBindGroup = nextActiveBlockDispatchArgsReadBindGroup;
        this.uniformsManager = uniformsManager;
        this.nParticleWorkgroups = nParticleWorkgroups;
        this.nBukkitWorkgroups = nBukkitWorkgroups;
        this.nActiveBlockWorkgroups = nActiveBlockWorkgroups;
        this.fusedSparseGridBindGroups = fusedSparseGridBindGroups;
        this.fusedActiveBlockDispatchArgsReadBindGroups = fusedActiveBlockDispatchArgsReadBindGroups;
        this.fusedBukkitBindGroups = fusedBukkitBindGroups;
        this.fusedG2p2gBindGroups = fusedG2p2gBindGroups;
        this.fusedActiveBlockDispatchBuffers = fusedActiveBlockDispatchBuffers;
        this.bukkitDispatchBuffer = bukkitDispatchBuffer;
        this.bukkitParticleReadBindGroups = bukkitParticleReadBindGroups;
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
        this.fusedActiveBlocksPrepared = false;
        this.fusedSourceSparseGridIndex = 0;
        this.dispatchMode = null;
    }

    private prepareClassicMode() {
        if (this.dispatchMode === "classic") return;

        this.dispatchMode = "classic";
        this.activeBlocksPrepared = false;
    }

    private prepareFusedMode() {
        if (this.dispatchMode === "fused") return;

        this.dispatchMode = "fused";
        this.fusedActiveBlocksPrepared = false;
        this.fusedSourceSparseGridIndex = 0;
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

    private encodeFusedSparseGridResetDispatch(
        computePassEncoder: GPUComputePassEncoder,
        sparseGridIndex: number,
    ) {
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.fusedSparseGridBindGroups[sparseGridIndex]);

        if (this.fusedBukkitGenerations[sparseGridIndex] >= BUKKIT_GENERATION_RESERVED - 1) {
            this.fusedBukkitGenerations[sparseGridIndex] = 1;
            computePassEncoder.setPipeline(this.resetBukkitTablePipeline);
            computePassEncoder.dispatchWorkgroups(this.nBukkitWorkgroups);
            return;
        }

        this.fusedBukkitGenerations[sparseGridIndex] += 1;
        computePassEncoder.setPipeline(this.clearBukkitPipeline);
        computePassEncoder.dispatchWorkgroups(1);
    }

    private bindFusedBukkitBuildGroups(
        computePassEncoder: GPUComputePassEncoder,
        sourceSparseGridIndex: number,
        bukkitBindGroupIndex: number,
    ) {
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.fusedSparseGridBindGroups[sourceSparseGridIndex]);
        computePassEncoder.setBindGroup(2, this.bukkitParticleReadBindGroups[sourceSparseGridIndex]);
        computePassEncoder.setBindGroup(3, this.fusedBukkitBindGroups[bukkitBindGroupIndex]);
    }

    private encodeBukkitBuildDispatches(
        computePassEncoder: GPUComputePassEncoder,
        sourceSparseGridIndex: number,
        bukkitBindGroupIndex: number,
    ) {
        this.bindFusedBukkitBuildGroups(computePassEncoder, sourceSparseGridIndex, bukkitBindGroupIndex);

        computePassEncoder.setPipeline(this.resetBukkitBuildBuffersPipeline);
        computePassEncoder.dispatchWorkgroups(this.nActiveBlockWorkgroups);

        computePassEncoder.setPipeline(this.countParticlesPerBukkitPipeline);
        computePassEncoder.dispatchWorkgroups(this.nParticleWorkgroups);

        computePassEncoder.setPipeline(this.allocateBukkitThreadDataPipeline);
        computePassEncoder.dispatchWorkgroups(this.nActiveBlockWorkgroups);

        computePassEncoder.setPipeline(this.insertParticlesIntoBukkitPipeline);
        computePassEncoder.dispatchWorkgroups(this.nParticleWorkgroups);

        computePassEncoder.setPipeline(this.finalizeBukkitDispatchPipeline);
        computePassEncoder.dispatchWorkgroups(1);
    }

    private encodeFusedBootstrapDispatches(
        computePassEncoder: GPUComputePassEncoder,
        enableInteraction: boolean,
    ) {
        const sourceSparseGridIndex = this.fusedSourceSparseGridIndex;
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.fusedSparseGridBindGroups[sourceSparseGridIndex]);

        this.encodeFusedSparseGridResetDispatch(computePassEncoder, sourceSparseGridIndex);

        computePassEncoder.setPipeline(this.mapAffectedBlocksPipeline);
        computePassEncoder.setBindGroup(2, this.particleReadBindGroup);
        computePassEncoder.dispatchWorkgroups(this.nParticleWorkgroups);

        computePassEncoder.setPipeline(this.mlsP2gComputePipeline);
        computePassEncoder.dispatchWorkgroups(this.nParticleWorkgroups);

        computePassEncoder.setPipeline(enableInteraction
            ? this.interactionGridComputePipeline
            : this.gridComputePipeline);
        computePassEncoder.setBindGroup(2, this.fusedActiveBlockDispatchArgsReadBindGroups[sourceSparseGridIndex]);
        computePassEncoder.dispatchWorkgroupsIndirect(
            this.fusedActiveBlockDispatchBuffers[sourceSparseGridIndex],
            0,
        );

        this.fusedActiveBlocksPrepared = true;
    }

    private encodeFusedBootstrapMapAndP2gDispatches(
        computePassEncoder: GPUComputePassEncoder,
    ) {
        const sourceSparseGridIndex = this.fusedSourceSparseGridIndex;
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.fusedSparseGridBindGroups[sourceSparseGridIndex]);

        this.encodeFusedSparseGridResetDispatch(computePassEncoder, sourceSparseGridIndex);

        computePassEncoder.setPipeline(this.mapAffectedBlocksPipeline);
        computePassEncoder.setBindGroup(2, this.particleReadBindGroup);
        computePassEncoder.dispatchWorkgroups(this.nParticleWorkgroups);

        computePassEncoder.setPipeline(this.mlsP2gComputePipeline);
        computePassEncoder.dispatchWorkgroups(this.nParticleWorkgroups);
    }

    private encodeFusedGridUpdateDispatch(
        computePassEncoder: GPUComputePassEncoder,
        sparseGridIndex: number,
        enableInteraction: boolean,
    ) {
        computePassEncoder.setPipeline(enableInteraction
            ? this.interactionGridComputePipeline
            : this.gridComputePipeline);
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.fusedSparseGridBindGroups[sparseGridIndex]);
        computePassEncoder.setBindGroup(2, this.fusedActiveBlockDispatchArgsReadBindGroups[sparseGridIndex]);
        computePassEncoder.dispatchWorkgroupsIndirect(
            this.fusedActiveBlockDispatchBuffers[sparseGridIndex],
            0,
        );
    }

    private encodeFusedMlsMpmBuildBukkitsPass(
        computePassEncoder: GPUComputePassEncoder,
        sourceSparseGridIndex: number,
        nextSparseGridIndex: number,
    ) {
        this.encodeBukkitBuildDispatches(
            computePassEncoder,
            sourceSparseGridIndex,
            nextSparseGridIndex,
        );
    }

    private encodeFusedMlsMpmG2p2gPass(
        computePassEncoder: GPUComputePassEncoder,
        sourceSparseGridIndex: number,
        nextSparseGridIndex: number,
        recordParticleSpeed: boolean,
    ) {
        this.encodeFusedSparseGridResetDispatch(computePassEncoder, nextSparseGridIndex);

        computePassEncoder.setPipeline(recordParticleSpeed
            ? this.fusedMlsSpeedRecordingG2p2gPipeline
            : this.fusedMlsG2p2gPipeline);
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.fusedSparseGridBindGroups[sourceSparseGridIndex]);
        computePassEncoder.setBindGroup(2, this.particleDataBindGroup);
        computePassEncoder.setBindGroup(3, this.fusedG2p2gBindGroups[nextSparseGridIndex]);
        computePassEncoder.dispatchWorkgroupsIndirect(this.bukkitDispatchBuffer, 0);

        computePassEncoder.setPipeline(this.finalizeNextActiveBlockDispatchPipeline);
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.fusedSparseGridBindGroups[sourceSparseGridIndex]);
        computePassEncoder.setBindGroup(2, this.bukkitParticleReadBindGroups[sourceSparseGridIndex]);
        computePassEncoder.setBindGroup(3, this.fusedBukkitBindGroups[nextSparseGridIndex]);
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
        this.prepareClassicMode();

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
        this.prepareClassicMode();

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
        let particleReadBindGroupBound = false;
        if (!this.activeBlocksPrepared) {
            this.encodeBukkitResetDispatch(computePassEncoder);

            computePassEncoder.setPipeline(this.mapAffectedBlocksPipeline);
            computePassEncoder.setBindGroup(2, this.particleReadBindGroup);
            particleReadBindGroupBound = true;
            computePassEncoder.dispatchWorkgroups(this.nParticleWorkgroups);
        }

        // particle-to-grid

        computePassEncoder.setPipeline(p2gPipeline);
        if (!particleReadBindGroupBound) {
            computePassEncoder.setBindGroup(2, this.particleReadBindGroup);
        }
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

    addFusedMlsMpmDispatchesBatch({
        computePassEncoder,
        enableInteraction,
        nSimulationSteps,
        recordParticleSpeed,
        bindCommonGroups = true,
    }: {
        computePassEncoder: GPUComputePassEncoder,
        enableInteraction: boolean,
        nSimulationSteps: number,
        recordParticleSpeed: boolean,
        bindCommonGroups?: boolean,
    }) {
        if (nSimulationSteps <= 0) return;
        this.prepareFusedMode();

        if (bindCommonGroups) {
            computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        }

        if (!this.fusedActiveBlocksPrepared) {
            this.encodeFusedBootstrapDispatches(computePassEncoder, enableInteraction);
        }

        const normalStepCount = recordParticleSpeed
            ? nSimulationSteps - 1
            : nSimulationSteps;

        for (let i = 0; i < normalStepCount; i++) {
            this.encodeFusedMlsMpmDispatch(computePassEncoder, enableInteraction, false);
        }

        if (recordParticleSpeed) {
            this.encodeFusedMlsMpmDispatch(computePassEncoder, enableInteraction, true);
        }
    }

    addFusedMlsMpmDispatchPassesBatch({
        commandEncoder,
        enableInteraction,
        nSimulationSteps,
        recordParticleSpeed,
        timestampWrites,
    }: {
        commandEncoder: GPUCommandEncoder,
        enableInteraction: boolean,
        nSimulationSteps: number,
        recordParticleSpeed: boolean,
        timestampWrites?: {
            querySet: GPUQuerySet,
            beginningOfPassWriteIndex: number,
            endOfPassWriteIndex: number,
        },
    }) {
        if (nSimulationSteps <= 0) return;
        this.prepareFusedMode();

        if (timestampWrites !== undefined) {
            const computePassEncoder = commandEncoder.beginComputePass({
                label: "fused MLS timestamp begin compute pass",
                timestampWrites: {
                    querySet: timestampWrites.querySet,
                    beginningOfPassWriteIndex: timestampWrites.beginningOfPassWriteIndex,
                },
            });
            computePassEncoder.end();
        }

        if (!this.fusedActiveBlocksPrepared) {
            {
                const computePassEncoder = commandEncoder.beginComputePass({
                    label: "fused MLS bootstrap map/P2G compute pass",
                });
                this.encodeFusedBootstrapMapAndP2gDispatches(computePassEncoder);
                computePassEncoder.end();
            }

            {
                const computePassEncoder = commandEncoder.beginComputePass({
                    label: "fused MLS bootstrap grid update compute pass",
                });
                this.encodeFusedGridUpdateDispatch(
                    computePassEncoder,
                    this.fusedSourceSparseGridIndex,
                    enableInteraction,
                );
                computePassEncoder.end();
            }

            this.fusedActiveBlocksPrepared = true;
        }

        const normalStepCount = recordParticleSpeed
            ? nSimulationSteps - 1
            : nSimulationSteps;

        for (let i = 0; i < normalStepCount; i++) {
            this.encodeFusedMlsMpmDispatchPasses(commandEncoder, enableInteraction, false);
        }

        if (recordParticleSpeed) {
            this.encodeFusedMlsMpmDispatchPasses(commandEncoder, enableInteraction, true);
        }

        if (timestampWrites !== undefined) {
            const computePassEncoder = commandEncoder.beginComputePass({
                label: "fused MLS timestamp end compute pass",
                timestampWrites: {
                    querySet: timestampWrites.querySet,
                    endOfPassWriteIndex: timestampWrites.endOfPassWriteIndex,
                },
            });
            computePassEncoder.end();
        }
    }

    private encodeFusedMlsMpmDispatchPasses(
        commandEncoder: GPUCommandEncoder,
        enableInteraction: boolean,
        recordParticleSpeed: boolean,
    ) {
        const sourceSparseGridIndex = this.fusedSourceSparseGridIndex;
        const nextSparseGridIndex = 1 - sourceSparseGridIndex;

        {
            const computePassEncoder = commandEncoder.beginComputePass({
                label: "fused MLS bukkit build compute pass",
            });
            this.encodeFusedMlsMpmBuildBukkitsPass(
                computePassEncoder,
                sourceSparseGridIndex,
                nextSparseGridIndex,
            );
            computePassEncoder.end();
        }

        {
            const computePassEncoder = commandEncoder.beginComputePass({
                label: "fused MLS G2P2G compute pass",
            });
            this.encodeFusedMlsMpmG2p2gPass(
                computePassEncoder,
                sourceSparseGridIndex,
                nextSparseGridIndex,
                recordParticleSpeed,
            );
            computePassEncoder.end();
        }

        {
            const computePassEncoder = commandEncoder.beginComputePass({
                label: "fused MLS grid update compute pass",
            });
            this.encodeFusedGridUpdateDispatch(
                computePassEncoder,
                nextSparseGridIndex,
                enableInteraction,
            );
            computePassEncoder.end();
        }

        this.fusedSourceSparseGridIndex = nextSparseGridIndex;
    }

    private encodeFusedMlsMpmDispatch(
        computePassEncoder: GPUComputePassEncoder,
        enableInteraction: boolean,
        recordParticleSpeed: boolean,
    ) {
        const sourceSparseGridIndex = this.fusedSourceSparseGridIndex;
        const nextSparseGridIndex = 1 - sourceSparseGridIndex;

        this.encodeBukkitBuildDispatches(
            computePassEncoder,
            sourceSparseGridIndex,
            nextSparseGridIndex,
        );

        this.encodeFusedSparseGridResetDispatch(computePassEncoder, nextSparseGridIndex);

        computePassEncoder.setPipeline(recordParticleSpeed
            ? this.fusedMlsSpeedRecordingG2p2gPipeline
            : this.fusedMlsG2p2gPipeline);
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.fusedSparseGridBindGroups[sourceSparseGridIndex]);
        computePassEncoder.setBindGroup(2, this.particleDataBindGroup);
        computePassEncoder.setBindGroup(3, this.fusedBukkitBindGroups[nextSparseGridIndex]);
        computePassEncoder.dispatchWorkgroupsIndirect(this.bukkitDispatchBuffer, 0);

        computePassEncoder.setPipeline(this.finalizeNextActiveBlockDispatchPipeline);
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.fusedSparseGridBindGroups[sourceSparseGridIndex]);
        computePassEncoder.setBindGroup(2, this.bukkitParticleReadBindGroups[sourceSparseGridIndex]);
        computePassEncoder.setBindGroup(3, this.fusedBukkitBindGroups[nextSparseGridIndex]);
        computePassEncoder.dispatchWorkgroups(1);

        computePassEncoder.setPipeline(enableInteraction
            ? this.interactionGridComputePipeline
            : this.gridComputePipeline);
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.fusedSparseGridBindGroups[nextSparseGridIndex]);
        computePassEncoder.setBindGroup(2, this.fusedActiveBlockDispatchArgsReadBindGroups[nextSparseGridIndex]);
        computePassEncoder.dispatchWorkgroupsIndirect(
            this.fusedActiveBlockDispatchBuffers[nextSparseGridIndex],
            0,
        );

        this.fusedSourceSparseGridIndex = nextSparseGridIndex;
    }
}
