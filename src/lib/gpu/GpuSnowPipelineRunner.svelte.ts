import { mat4, type Mat4 } from "wgpu-matrix";
import type { Camera } from "$lib/components/simulationViewer/Camera.svelte";
import { GpuPointsRenderPipelineManager } from "./pointsRender/GpuPointsRenderPipelineManager";
import { GpuMpmPipelineManager } from "./mpm/GpuMpmPipelineManager";
import { GpuUniformsBufferManager } from "./uniforms/GpuUniformsBufferManager";
import { GpuMpmBufferManager } from "./mpm/GpuMpmBufferManager";
import { GpuRenderMethodType, type GpuRenderMethod } from "./GpuRenderMethod";
import { GpuPerformanceMeasurementBufferManager } from "./performanceMeasurement/GpuPerformanceMeasurementBufferManager";
import { GpuSpawnVolumeBufferManager, type SpawnPointSource } from "./particleInitialize/GpuSpawnVolumeBufferManager";
import { GpuColliderBufferManager } from "./collider/GpuColliderBufferManager";
import { GpuParticleInitializePipelineManager } from "./particleInitialize/GpuParticleInitializePipelineManager";
import { GpuRasterizeRenderPipelineManager } from "./collider/GpuRasterizeRenderPipelineManager";
import { GpuMpmGridRenderPipelineManager } from "./mpmGridRender/GpuMpmGridRenderPipelineMager";
import { GpuVolumetricBufferManager } from "./volumetric/GpuVolumetricBufferManager";
import { GpuVolumetricRenderPipelineManager } from "./volumetric/GpuVolumetricRenderPipelineManager";
import { GpuRaymarchingSurfaceRenderPipelineManager } from "./raymarching/GpuRaymarchingSurfaceRenderPipelineManager";
import { GpuSsfrRenderPipelineManager } from "./ssfr/GpuSsfrRenderPipelineManager";
import { GpuMarchingCubesRenderPipelineManager } from "./marchingCubes/GpuMarchingCubesRenderPipelineManager";
import { GpuSplatsRenderPipelineManager } from "./splats/GpuSplatsRenderPipelineManager";
import type { ColliderGeometry } from "./collider/GpuColliderBufferManager";
import { GpuSimulationMethodType } from "./GpuSimulationMethod";
import { GpuEnvironmentRenderPipelineManager } from "./environmentMap/GpuEnvironmentRenderPipelineManager";
import { GpuEnvironmentTextureManager } from "./environmentMap/GpuEnvironmentTextureManager";
import { untrack } from "svelte";
import { PrerenderPassElapsedTime } from "$lib/components/simulationViewer/PrerenderPassElapsedTime.svelte";
import { GpuSsaoPipelineManager } from "./ssao/GpuSsaoPipelineManager";
import { GpuDepthPicker } from "./GpuDepthPicker";
import { GpuParticleSpeedReductionPipelineManager } from "./mpm/GpuParticleSpeedReductionPipelineManager";
import { GpuParticleAppearanceBufferManager } from "./particleAppearance/GpuParticleAppearanceBufferManager";
import { GRAVITATIONAL_ACCELERATION_M_PER_S2 } from "./gravity";
import { GpuSimulationPlaybackFrameCacheManager } from "./simulationFrameCache/GpuSimulationPlaybackFrameCacheManager";
import {
    CFL_NUMBER,
    calculateSimulationFrameSchedule,
    calculateCflLimitedSimulationTimestepS,
    calculateSimulationSubstepTimestepS,
    calculateSimulationSubstepsPerMaxStep,
    canRelaxParticleSpeedSampling,
    calculateSpawnSourceMaxElasticWaveSpeed,
    MAX_SIMULATION_DRIFT_MS,
    MAX_SIMULATION_STEPS_PER_FRAME,
    MAX_SIMULATION_SUBSTEPS_PER_FRAME,
    MIN_SIMULATION_TIMESTEP_S,
} from "./simulationTimestep";

export type GpuFrameTiming = {
    computeSimulationStepNs: bigint,
    computeSimulationSubstepNs: bigint,
    nSimulationSubsteps: number,
    renderNs: bigint,
    postprocessRenderNs: bigint,
};

export type GpuFrameTimingCallbacks = {
    onGpuTimeUpdate?: (times: GpuFrameTiming) => void,
    onAnimationFrameTimeUpdate?: (ms: number) => void,
};

export type GpuStillFrameRenderOptions = GpuFrameTimingCallbacks & {
    measureGpuTimestamps?: boolean,
};

export type GpuFixedSimulationStepResult = {
    nSimulationSubsteps: number,
    simulationTimestepS: number,
    simulatedTimeS: number,
};

const FP_SCALE = 65536;
const MAX_FIXED_POINT_I32 = 2_147_483_000;
const MAX_FIXED_POINT_GRID_SPEED = MAX_FIXED_POINT_I32 / FP_SCALE;
const GPU_TIMING_SAMPLE_INTERVAL_FRAMES = 7;
const PARTICLE_SPEED_SAMPLE_INTERVAL_FRAMES = 4;
// A prime-ish relaxed cadence avoids regular aliasing with browser frame pacing.
// Due speed samples preempt GPU timestamp telemetry in the frame loop below.
const PARTICLE_SPEED_RELAXED_SAMPLE_INTERVAL_FRAMES = 23;
const PARTICLE_SPEED_RELAXED_CFL_HEADROOM = 0.8;
const MOVING_COLLIDER_SPEED_EPSILON = 1e-3;
const COLLIDER_VELOCITY_ZERO_EPSILON_SQUARED = 1e-20;
const COLLIDER_TRANSFORM_IDENTITY_EPSILON = 1e-6;
const COLLIDER_SDF_LAST_COORD = 63;

const IDENTITY_MAT4_VALUES = [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
];

const mat4IsIdentity = (matrix: Mat4) => {
    for (let i = 0; i < IDENTITY_MAT4_VALUES.length; i++) {
        if (Math.abs(matrix[i] - IDENTITY_MAT4_VALUES[i]) > COLLIDER_TRANSFORM_IDENTITY_EPSILON) {
            return false;
        }
    }

    return true;
};

const emptyColliderGeometry = (): ColliderGeometry => ({
    positions: [],
    normals: [],
    uvs: [],
    materialIndices: [],
    textures: [],
    indices: [],
    objects: [],
});

const boundsFromSpawnSource = (source: SpawnPointSource): {
    min: [number, number, number],
    max: [number, number, number],
} => {
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];

    if (source.type === "mesh") {
        for (const vertex of source.vertices) {
            min[0] = Math.min(min[0], vertex[0]);
            min[1] = Math.min(min[1], vertex[1]);
            min[2] = Math.min(min[2], vertex[2]);
            max[0] = Math.max(max[0], vertex[0]);
            max[1] = Math.max(max[1], vertex[1]);
            max[2] = Math.max(max[2], vertex[2]);
        }
    } else {
        for (let i = 0; i < source.points.length; i += 4) {
            min[0] = Math.min(min[0], source.points[i]);
            min[1] = Math.min(min[1], source.points[i + 1]);
            min[2] = Math.min(min[2], source.points[i + 2]);
            max[0] = Math.max(max[0], source.points[i]);
            max[1] = Math.max(max[1], source.points[i + 1]);
            max[2] = Math.max(max[2], source.points[i + 2]);
        }
    }

    if (
        !min.every(Number.isFinite)
        || !max.every(Number.isFinite)
    ) {
        return {
            min: [-5, -5, -5],
            max: [5, 5, 5],
        };
    }

    return { min, max };
};

export class GpuSnowPipelineRunner {
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly nParticles: number;
    private readonly explicitMpmMaxSimulationTimestepS: () => number;
    private readonly mlsMpmMaxSimulationTimestepS: () => number;
    private readonly camera: Camera;
    private readonly colliderFriction: () => number;
    private readonly isInteracting: () => boolean;
    private readonly interactionStrength: () => number;
    private readonly minGridCellDim: number;
    private readonly elasticWaveSpeed: number;
    private readonly colliderMinCoords: [number, number, number];
    private readonly colliderMaxCoords: [number, number, number];
    private colliderSdfMaxCellSize = 1;
    private depthTextureView: GPUTextureView;

    readonly uniformsManager: GpuUniformsBufferManager;
    private readonly performanceMeasurementManager: GpuPerformanceMeasurementBufferManager | null;
    private readonly mpmPipelineManager: GpuMpmPipelineManager;
    private readonly particleSpeedReductionPipelineManager: GpuParticleSpeedReductionPipelineManager;
    private readonly simulationPlaybackFrameCacheManager: GpuSimulationPlaybackFrameCacheManager;
    private readonly rasterizeRenderPipelineManager: GpuRasterizeRenderPipelineManager | null;
    private readonly mpmGridRenderPipelineManager: GpuMpmGridRenderPipelineManager;
    private readonly particleInitializePipelineManager: GpuParticleInitializePipelineManager;
    private readonly environmentRenderPipelineManager: GpuEnvironmentRenderPipelineManager;
    private readonly ssaoPipelineManager: GpuSsaoPipelineManager;
    private readonly depthPicker: GpuDepthPicker;

    private depthTexture: GPUTexture | null = null;
    private stopEffects: (() => void) | null = null;
    private destroyed = false;

    private renderMethod = $state<GpuRenderMethod | null>(null);
    readonly prerenderPasses = $derived(this.renderMethod?.prerenderPasses() ?? null);
    private renderHasPrerenderPasses = false;
    private renderUsesSsao = false;
    #prerenderElapsedTimes = $state<PrerenderPassElapsedTime[] | null>(null);
    readonly prerenderElapsedTimes = $derived(this.#prerenderElapsedTimes);
    private latestMaxParticleSpeed = $state(0);
    private currentColliderSpeed = $state(0);
    private colliderTransformIsIdentity = true;
    private colliderVelocityIsZero = true;
    private colliderSdfValid = false;
    private lastWrittenSimulationTimestepS = Number.NaN;
    private lastWrittenMaxStableParticleSpeed = Number.NaN;


    private readonly mpmManager: GpuMpmBufferManager;
    private readonly spawnVolumeManager: GpuSpawnVolumeBufferManager;
    private readonly particleAppearanceManager: GpuParticleAppearanceBufferManager;
    private readonly colliderManager: GpuColliderBufferManager;
    private readonly environmentTextureManager: GpuEnvironmentTextureManager;

    private readonly measurePerf: boolean;
    // debug
    // private readonly readbackBuffer : GPUBuffer;
    // v : [number, number, number] = [0.0, 0.0, 0.0];

    private readonly getSimulationMethodType: () => GpuSimulationMethodType;
    private readonly getRenderMethodType: () => GpuRenderMethodType;
    private readonly oneSimulationStepPerFrame: () => boolean;

    get simulationPlaybackFrameLayout() {
        return this.simulationPlaybackFrameCacheManager.layout;
    }

    constructor({
        device,
        format,
        context,
        nParticles,
        gridResolutionX,
        gridResolutionY,
        gridResolutionZ,
        mcGridResolutionX,
        mcGridResolutionY,
        mcGridResolutionZ,
        explicitMpmMaxSimulationTimestepS,
        mlsMpmMaxSimulationTimestepS,
        camera,
        spawnSource,
        collider,
        particleAppearances,
        getSimulationMethodType,
        getRenderMethodType,
        oneSimulationStepPerFrame,
        environmentImageBitmap,
        measurePerf,
        width,
        height,
        colliderFriction,
        isInteracting,
        interactionStrength,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        context: GPUCanvasContext,
        nParticles: number,
        gridResolutionX: number,
        gridResolutionY: number,
        gridResolutionZ: number,
        mcGridResolutionX?: number,
        mcGridResolutionY?: number,
        mcGridResolutionZ?: number,
        explicitMpmMaxSimulationTimestepS: () => number,
        mlsMpmMaxSimulationTimestepS: () => number,
        camera: Camera,
        spawnSource: SpawnPointSource,
        collider?: ColliderGeometry | null,
        particleAppearances?: Uint32Array | null,
        getSimulationMethodType: () => GpuSimulationMethodType,
        getRenderMethodType: () => GpuRenderMethodType,
        oneSimulationStepPerFrame: () => boolean,
        environmentImageBitmap: ImageBitmap,
        measurePerf: boolean,
        width: () => number,
        height: () => number,
        colliderFriction: () => number,
        isInteracting: () => boolean,
        interactionStrength: () => number,
    }) {
        this.device = device;
        this.context = context;
        this.nParticles = nParticles;
        this.explicitMpmMaxSimulationTimestepS = explicitMpmMaxSimulationTimestepS;
        this.mlsMpmMaxSimulationTimestepS = mlsMpmMaxSimulationTimestepS;
        this.colliderFriction = colliderFriction;
        this.isInteracting = isInteracting;
        this.interactionStrength = interactionStrength;
        this.elasticWaveSpeed = calculateSpawnSourceMaxElasticWaveSpeed(spawnSource);

        this.camera = camera;

        this.depthTexture = device.createTexture({
            size: [camera.screenDims.width(), camera.screenDims.height()],
            format: "depth32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.depthTextureView = this.depthTexture.createView();

        const uniformsManager = new GpuUniformsBufferManager({device});
        this.uniformsManager = uniformsManager;

        const gridMinCoords: [number, number, number] = [-5, -5, -5];
        const gridMaxCoords: [number, number, number] = [5, 5, 5];
        const gridCellDims: [number, number, number] = [
            (gridMaxCoords[0] - gridMinCoords[0]) / gridResolutionX,
            (gridMaxCoords[1] - gridMinCoords[1]) / gridResolutionY,
            (gridMaxCoords[2] - gridMinCoords[2]) / gridResolutionZ,
        ];
        const invGridCellDims: [number, number, number] = [
            1 / gridCellDims[0],
            1 / gridCellDims[1],
            1 / gridCellDims[2],
        ];
        const invGridCellDimsSquared: [number, number, number] = [
            invGridCellDims[0] * invGridCellDims[0],
            invGridCellDims[1] * invGridCellDims[1],
            invGridCellDims[2] * invGridCellDims[2],
        ];
        const simulationDomainCenter: [number, number, number] = [
            0.5 * (gridMinCoords[0] + gridMaxCoords[0]),
            0.5 * (gridMinCoords[1] + gridMaxCoords[1]),
            0.5 * (gridMinCoords[2] + gridMaxCoords[2]),
        ];
        const simulationDomainMaxInside: [number, number, number] = [
            gridMaxCoords[0] - gridCellDims[0] * 0.001,
            gridMaxCoords[1] - gridCellDims[1] * 0.001,
            gridMaxCoords[2] - gridCellDims[2] * 0.001,
        ];
        this.minGridCellDim = Math.min(...gridCellDims);

        uniformsManager.writeGridResolution([gridResolutionX, gridResolutionY, gridResolutionZ]);
        uniformsManager.writeGridCellDims(gridCellDims);
        uniformsManager.writeInvGridCellDims(invGridCellDims);
        uniformsManager.writeInvGridCellDimsSquared(invGridCellDimsSquared);
        uniformsManager.writeSimulationDomainDerivedValues({
            center: simulationDomainCenter,
            maxInside: simulationDomainMaxInside,
        });
        uniformsManager.writeFixedPointScale(FP_SCALE);
        uniformsManager.writeGridMinCoords(gridMinCoords);
        uniformsManager.writeGridMaxCoords(gridMaxCoords);

        this.performanceMeasurementManager = measurePerf
            ? new GpuPerformanceMeasurementBufferManager({device})
            : null;

        this.measurePerf = measurePerf;

        const mpmManager = new GpuMpmBufferManager({
            device,
            nParticles,
        });
        this.mpmManager = mpmManager;

        this.particleSpeedReductionPipelineManager = new GpuParticleSpeedReductionPipelineManager({
            device,
        });
        this.simulationPlaybackFrameCacheManager = new GpuSimulationPlaybackFrameCacheManager({
            device,
            nParticles,
            particleDataBuffer: mpmManager.particleDataBuffer,
            particleFlagsBuffer: mpmManager.particleFlagsBuffer,
        });

        const spawnBounds = boundsFromSpawnSource(spawnSource);
        uniformsManager.writeMeshMinCoords(spawnBounds.min);
        uniformsManager.writeMeshMaxCoords(spawnBounds.max);

        const spawnVolumeManager = new GpuSpawnVolumeBufferManager({
            device,
            nParticles,
            source: spawnSource,
        });
        this.spawnVolumeManager = spawnVolumeManager;

        const particleAppearanceManager = new GpuParticleAppearanceBufferManager({
            device,
            nParticles,
            appearances: particleAppearances,
        });
        this.particleAppearanceManager = particleAppearanceManager;

        const colliderGeometry = collider ?? emptyColliderGeometry();
        const colliderManager = new GpuColliderBufferManager({
            device, 
            vertices: colliderGeometry.positions,
            normals: colliderGeometry.normals,
            uvs: colliderGeometry.uvs,
            materialIndices: colliderGeometry.materialIndices,
            textures: colliderGeometry.textures,
            indices: colliderGeometry.indices,
        });
        this.colliderManager = colliderManager;
        this.colliderMinCoords = colliderManager.minCoords;
        this.colliderMaxCoords = colliderManager.maxCoords;
        uniformsManager.writeColliderMinCoords(colliderManager.minCoords);
        uniformsManager.writeColliderMaxCoords(colliderManager.maxCoords);
        this.writeColliderSdfMetadata(colliderManager.minCoords, colliderManager.maxCoords);
        const colliderTransform = mat4.identity();
        uniformsManager.writeColliderTransformMat(colliderTransform);
        uniformsManager.writeColliderTransformInv(colliderTransform);
        uniformsManager.writeColliderTransformIsIdentity(true);
        this.writeColliderWorldBounds(colliderTransform);
        uniformsManager.writeColliderVel([0.0, 0.0, 0.0]);
        uniformsManager.writeColliderVelocityIsZero(true);

        // Compute
        const particleInitializePipelineManager = new GpuParticleInitializePipelineManager({
            device,
            particleDataBuffer: mpmManager.particleDataBuffer,
            particleFlagsBuffer: mpmManager.particleFlagsBuffer,
            spawnPointsBuffer: spawnVolumeManager.spawnPointsBuffer,
            uniformsManager,
        });
        this.particleInitializePipelineManager = particleInitializePipelineManager;


        const mpmPipelineManager = new GpuMpmPipelineManager({
            device,
            nParticles,
            gridResolutionX,
            gridResolutionY,
            gridResolutionZ,
            particleDataBuffer: mpmManager.particleDataBuffer,
            particleFlagsBuffer: mpmManager.particleFlagsBuffer,
            sparseGridBuffer: mpmManager.sparseGridBuffer,
            nextSparseGridBuffer: mpmManager.nextSparseGridBuffer,
            gridAccumulatorBuffer: mpmManager.gridAccumulatorBuffer,
            nextGridAccumulatorBuffer: mpmManager.nextGridAccumulatorBuffer,
            gridVelocityBuffer: mpmManager.gridVelocityBuffer,
            maxParticleSpeedBuffer: this.particleSpeedReductionPipelineManager.maxSpeedBuffer,
            activeBlockDispatchBuffer: mpmManager.activeBlockDispatchBuffer,
            nextActiveBlockDispatchBuffer: mpmManager.nextActiveBlockDispatchBuffer,
            bukkitParticleCountsBuffer: mpmManager.bukkitParticleCountsBuffer,
            bukkitInsertCountersBuffer: mpmManager.bukkitInsertCountersBuffer,
            bukkitIndexStartBuffer: mpmManager.bukkitIndexStartBuffer,
            bukkitThreadDataBuffer: mpmManager.bukkitThreadDataBuffer,
            bukkitParticleDataBuffer: mpmManager.bukkitParticleDataBuffer,
            bukkitDispatchBuffer: mpmManager.bukkitDispatchBuffer,
            bukkitParticleAllocatorBuffer: mpmManager.bukkitParticleAllocatorBuffer,
            bukkitThreadGroupCountBuffer: mpmManager.bukkitThreadGroupCountBuffer,
            uniformsManager,
            colliderManager,
        });

        this.mpmPipelineManager = mpmPipelineManager;

        // Render
        const rasterizeRenderPipeline = colliderGeometry.indices.length > 0
            ? new GpuRasterizeRenderPipelineManager({
                device,
                format,
                depthFormat: "depth32float",
                uniformsManager: uniformsManager,
                colliderManager: colliderManager,
            })
            : null;
        this.rasterizeRenderPipelineManager = rasterizeRenderPipeline;

        const mpmGridRenderPipelineManager = new GpuMpmGridRenderPipelineManager({
            device,
            format,
            depthFormat: "depth32float",
            uniformsManager,
            mpmManager,
        });
        this.mpmGridRenderPipelineManager = mpmGridRenderPipelineManager;


        const environmentTextureManager = new GpuEnvironmentTextureManager({
            device,
            imageBitmap: environmentImageBitmap,
        });
        this.environmentTextureManager = environmentTextureManager;

        const environmentRenderPipelineManager = new GpuEnvironmentRenderPipelineManager({
            device,
            uniformsManager,
            textureManager: environmentTextureManager,
            format,
            depthFormat: "depth32float",
        });
        this.environmentRenderPipelineManager = environmentRenderPipelineManager;

        this.ssaoPipelineManager = new GpuSsaoPipelineManager({
            device,
            format,
            uniformsManager,
        });

        this.depthPicker = new GpuDepthPicker({ device });

        this.getSimulationMethodType = getSimulationMethodType;
        this.getRenderMethodType = getRenderMethodType;
        this.oneSimulationStepPerFrame = oneSimulationStepPerFrame;

        this.stopEffects = $effect.root(() => {
            $effect(() => this.uniformsManager.writeViewProjMat(this.camera.viewProjMat));
            $effect(() => this.uniformsManager.writeViewProjInvMat(this.camera.viewProjInvMat));
            $effect(() => {
                const viewInv = this.camera.viewInvMat;
                this.uniformsManager.writeCameraPos([viewInv[12], viewInv[13], viewInv[14]]);
            });
            $effect(() => this.writeSimulationTimingUniforms(this.selectedSimulationTimestepS));
            $effect(() => this.uniformsManager.writeColliderFriction(this.colliderFriction()));


            let lastRenderMethodType: GpuRenderMethodType | null = null;
            $effect(() => {
                if (this.destroyed) return;

                const renderMethodType = getRenderMethodType();
                if (renderMethodType === lastRenderMethodType) return;

                this.renderMethod?.destroy();
                this.renderMethod = null;
                lastRenderMethodType = renderMethodType;




                switch (renderMethodType) {
                    case GpuRenderMethodType.Points:
                        this.renderMethod = new GpuPointsRenderPipelineManager({
                            device,
                            format,
                            depthFormat: "depth32float",
                            uniformsManager,
                            mpmManager,
                            particleAppearanceManager,
                        });
                        break;

                    case GpuRenderMethodType.Splats:
                        this.renderMethod = new GpuSplatsRenderPipelineManager({
                            device,
                            format,
                            depthFormat: "depth32float",
                            uniformsManager,
                            mpmManager,
                            particleAppearanceManager,
                        });
                        break;

                    case GpuRenderMethodType.Volumetric: {
                        const volumetricBufferManager = new GpuVolumetricBufferManager({
                            device,
                            gridResolutionX,
                            gridResolutionY,
                            gridResolutionZ,
                            screenDims: {
                                width: width(),
                                height: height(),
                            },
                        });

                        this.renderMethod = new GpuVolumetricRenderPipelineManager({
                            device,
                            format,
                            depthFormat: "depth32float",
                            uniformsManager,
                            volumetricBufferManager,
                            mpmBufferManager: mpmManager,
                            environmentTextureManager,
                            performanceMeasurementManager: this.performanceMeasurementManager,
                        });
                        break;
                    }

                    case GpuRenderMethodType.Ssfr:
                        this.renderMethod = new GpuSsfrRenderPipelineManager({
                            device,
                            format,
                            depthFormat: "depth32float",
                            uniformsManager,
                            mpmManager,
                            performanceMeasurementManager: this.performanceMeasurementManager,
                        });
                        break;

                    case GpuRenderMethodType.MarchingCubes:
                        this.renderMethod = new GpuMarchingCubesRenderPipelineManager({
                            device,
                            format,
                            depthFormat: "depth32float",
                            uniformsManager,
                            mpmManager,
                            gridResolutionX,
                            gridResolutionY,
                            gridResolutionZ,
                            mcGridResolutionX,
                            mcGridResolutionY,
                            mcGridResolutionZ,
                            performanceMeasurementManager: this.performanceMeasurementManager,
                        });
                        break;

                    case GpuRenderMethodType.RaymarchingSurface: {
                        const volumetricBufferManager = new GpuVolumetricBufferManager({
                            device,
                            gridResolutionX,
                            gridResolutionY,
                            gridResolutionZ,
                            screenDims: {
                                width: width(),
                                height: height(),
                            },
                        });

                        this.renderMethod = new GpuRaymarchingSurfaceRenderPipelineManager({
                            device,
                            format,
                            depthFormat: "depth32float",
                            uniformsManager,
                            volumetricBufferManager,
                            mpmBufferManager: mpmManager,
                            environmentTextureManager,
                            performanceMeasurementManager: this.performanceMeasurementManager,
                        });
                        break;
                    }
                }

                const prerenderPasses = this.renderMethod?.prerenderPasses() ?? [];
                this.renderHasPrerenderPasses = prerenderPasses.length > 0;
                this.renderUsesSsao = renderMethodType === GpuRenderMethodType.MarchingCubes
                    || renderMethodType === GpuRenderMethodType.Points
                    || renderMethodType === GpuRenderMethodType.Splats;
            });

            $effect(() => {
                if (this.destroyed) return;

                this.depthTexture?.destroy();

                this.depthTexture = this.device.createTexture({
                    size: [width(), height()],
                    format: "depth32float",
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.depthTextureView = this.depthTexture.createView();

                this.renderMethod?.resize(this.device, width(), height(), this.depthTextureView);
                this.ssaoPipelineManager.resize(this.depthTextureView);
            });

            $effect(() => {
                const prerenderPasses = this.prerenderPasses;
                if (prerenderPasses === null) return;

                untrack(() => {
                    this.#prerenderElapsedTimes = prerenderPasses.map(passLabel => new PrerenderPassElapsedTime(passLabel));
                });
            });


            return () => {
                this.renderMethod?.destroy();
                this.renderMethod = null;
            };
        });
    }

    destroy() {
        if (this.destroyed) return;

        this.destroyed = true;
        this.stopEffects?.();
        this.stopEffects = null;

        this.renderMethod?.destroy();
        this.renderMethod = null;
        this.depthTexture?.destroy();
        this.depthTexture = null;

        this.depthPicker.destroy();
        this.environmentRenderPipelineManager.destroy();
        this.environmentTextureManager.destroy();
        this.mpmGridRenderPipelineManager.destroy();
        this.simulationPlaybackFrameCacheManager.destroy();
        this.particleSpeedReductionPipelineManager.destroy();
        this.colliderManager.destroy();
        this.particleAppearanceManager.destroy();
        this.spawnVolumeManager.destroy();
        this.performanceMeasurementManager?.destroy();
        this.mpmManager.destroy();
        this.uniformsManager.destroy();
    }

    scatterParticles() {
        if (this.destroyed) return;

        this.latestMaxParticleSpeed = 0;
        this.writeSimulationTimingUniforms(this.selectedSimulationTimestepS);
        this.mpmPipelineManager.invalidateActiveBlocks();

        const commandEncoder = this.device.createCommandEncoder({
            label: "particle scatter command encoder",
        });

        this.particleInitializePipelineManager.addDispatch({
            commandEncoder,
            nParticles: this.nParticles,
        });

        this.device.queue.submit([commandEncoder.finish()]);
    }

    scatterParticlesInMeshVolume() {
        this.scatterParticles();
    }

    private frameTimestampMetadata() {
        const frameRenderUsesSsao = this.renderUsesSsao;
        const framePrerenderTimestampBaseIndex = frameRenderUsesSsao ? 6 : 4;
        const framePrerenderPassCount = this.renderHasPrerenderPasses
            ? (this.prerenderPasses?.length ?? 0)
            : 0;
        const timestampQueryCount = framePrerenderTimestampBaseIndex + 2 * framePrerenderPassCount;

        return {
            frameRenderUsesSsao,
            framePrerenderTimestampBaseIndex,
            timestampQueryCount,
        };
    }

    private canMeasureGpuTimestamps() {
        return this.performanceMeasurementManager !== null
            && this.renderMethod !== null
            && this.performanceMeasurementManager.canScheduleReadback();
    }

    private addNoopComputeTimestampPass(commandEncoder: GPUCommandEncoder) {
        if (this.performanceMeasurementManager === null) return;

        const computePassEncoder = commandEncoder.beginComputePass({
            label: "still frame timestamp compute pass",
            timestampWrites: {
                querySet: this.performanceMeasurementManager.querySet,
                beginningOfPassWriteIndex: 0,
                endOfPassWriteIndex: 1,
            },
        });
        computePassEncoder.end();
    }

    private mapSubmittedGpuTimes({
        nSimulationSubsteps,
        frameRenderUsesSsao,
        framePrerenderTimestampBaseIndex,
        onGpuTimeUpdate,
        onError,
    }: {
        nSimulationSubsteps: number,
        frameRenderUsesSsao: boolean,
        framePrerenderTimestampBaseIndex: number,
        onGpuTimeUpdate?: (times: GpuFrameTiming) => void,
        onError?: (error: unknown) => void,
    }) {
        if (this.performanceMeasurementManager === null) return;

        this.performanceMeasurementManager.mapTime(timestamps => {
            const computeSimulationStepNs = timestamps[1] - timestamps[0];
            const computeSimulationSubstepNs = nSimulationSubsteps > 0
                ? computeSimulationStepNs / BigInt(nSimulationSubsteps)
                : 0n;
            const renderNs = timestamps[3] - timestamps[2];
            const postprocessRenderNs = frameRenderUsesSsao
                ? timestamps[5] - timestamps[4]
                : 0n;

            if (this.#prerenderElapsedTimes !== null) {
                for (let i = 0; i < this.#prerenderElapsedTimes.length; i++) {
                    const index = framePrerenderTimestampBaseIndex + 2 * i;
                    this.#prerenderElapsedTimes[i].elapsedTimeNs = timestamps[index + 1] - timestamps[index];
                }
            }

            onGpuTimeUpdate?.({
                computeSimulationStepNs,
                computeSimulationSubstepNs,
                nSimulationSubsteps,
                renderNs,
                postprocessRenderNs,
            });
        })
            .catch(error => {
                console.error(error);
                onError?.(error);
            });
    }

    async readSimulationPlaybackFrame() {
        if (this.destroyed) {
            throw new Error("cannot read a destroyed simulation");
        }

        const { byteLength } = this.simulationPlaybackFrameLayout;
        const readbackBuffer = this.device.createBuffer({
            label: "simulation playback frame readback buffer",
            size: byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const commandEncoder = this.device.createCommandEncoder({
            label: "simulation playback frame readback command encoder",
        });

        this.simulationPlaybackFrameCacheManager.addPackDispatch({ commandEncoder });
        commandEncoder.copyBufferToBuffer(
            this.simulationPlaybackFrameCacheManager.frameBuffer,
            0,
            readbackBuffer,
            0,
            byteLength,
        );

        this.device.queue.submit([commandEncoder.finish()]);

        let mapped = false;
        try {
            await readbackBuffer.mapAsync(GPUMapMode.READ);
            mapped = true;

            if (this.destroyed) {
                throw new Error("simulation was destroyed before playback frame readback completed");
            }

            return readbackBuffer.getMappedRange().slice(0);
        } finally {
            if (mapped && readbackBuffer.mapState === "mapped") {
                readbackBuffer.unmap();
            }
            readbackBuffer.destroy();
        }
    }

    restoreSimulationPlaybackFrame(frame: ArrayBuffer) {
        if (this.destroyed) return;

        this.simulationPlaybackFrameCacheManager.writeFrame(frame);

        const commandEncoder = this.device.createCommandEncoder({
            label: "simulation playback frame restore command encoder",
        });
        this.simulationPlaybackFrameCacheManager.addRestoreDispatch({ commandEncoder });
        this.device.queue.submit([commandEncoder.finish()]);

        this.latestMaxParticleSpeed = 0;
        this.mpmPipelineManager.invalidateActiveBlocks();
    }

    renderSimulationPlaybackFrame(
        frame: ArrayBuffer,
        {
            onGpuTimeUpdate,
            onAnimationFrameTimeUpdate,
            measureGpuTimestamps = false,
        }: GpuStillFrameRenderOptions = {},
    ) {
        if (this.destroyed) return;

        const frameStartMs = performance.now();
        this.simulationPlaybackFrameCacheManager.writeFrame(frame);

        const shouldMeasureGpuTimestamps = measureGpuTimestamps && this.canMeasureGpuTimestamps();
        this.performanceMeasurementManager?.setEnabled(shouldMeasureGpuTimestamps);
        const {
            frameRenderUsesSsao,
            framePrerenderTimestampBaseIndex,
            timestampQueryCount,
        } = this.frameTimestampMetadata();
        const commandEncoder = this.device.createCommandEncoder({
            label: "simulation playback frame restore and render command encoder",
        });

        this.simulationPlaybackFrameCacheManager.addRestoreDispatch({ commandEncoder });

        if (shouldMeasureGpuTimestamps) {
            this.addNoopComputeTimestampPass(commandEncoder);
        }

        this.addRender(commandEncoder, shouldMeasureGpuTimestamps);

        if (shouldMeasureGpuTimestamps && this.performanceMeasurementManager !== null) {
            this.performanceMeasurementManager.addResolve(commandEncoder, timestampQueryCount);
        }

        this.device.queue.submit([commandEncoder.finish()]);
        this.latestMaxParticleSpeed = 0;
        this.mpmPipelineManager.invalidateActiveBlocks();
        onAnimationFrameTimeUpdate?.(performance.now() - frameStartMs);

        if (shouldMeasureGpuTimestamps) {
            this.mapSubmittedGpuTimes({
                nSimulationSubsteps: 0,
                frameRenderUsesSsao,
                framePrerenderTimestampBaseIndex,
                onGpuTimeUpdate,
            });
        }
    }

    renderStillFrame({
        onGpuTimeUpdate,
        onAnimationFrameTimeUpdate,
        measureGpuTimestamps = true,
    }: GpuStillFrameRenderOptions = {}) {
        if (this.destroyed) return;

        const frameStartMs = performance.now();
        const shouldMeasureGpuTimestamps = measureGpuTimestamps && this.canMeasureGpuTimestamps();
        this.performanceMeasurementManager?.setEnabled(shouldMeasureGpuTimestamps);
        const {
            frameRenderUsesSsao,
            framePrerenderTimestampBaseIndex,
            timestampQueryCount,
        } = this.frameTimestampMetadata();
        const commandEncoder = this.device.createCommandEncoder({
            label: "still frame render command encoder",
        });

        if (shouldMeasureGpuTimestamps) {
            this.addNoopComputeTimestampPass(commandEncoder);
        }

        this.addRender(commandEncoder, shouldMeasureGpuTimestamps);

        if (shouldMeasureGpuTimestamps && this.performanceMeasurementManager !== null) {
            this.performanceMeasurementManager.addResolve(commandEncoder, timestampQueryCount);
        }

        this.device.queue.submit([commandEncoder.finish()]);
        onAnimationFrameTimeUpdate?.(performance.now() - frameStartMs);

        if (shouldMeasureGpuTimestamps) {
            this.mapSubmittedGpuTimes({
                nSimulationSubsteps: 0,
                frameRenderUsesSsao,
                framePrerenderTimestampBaseIndex,
                onGpuTimeUpdate,
            });
        }
    }

    advanceFixedSimulationSubsteps({
        nSubsteps,
        onGpuTimeUpdate,
        onAnimationFrameTimeUpdate,
    }: GpuFrameTimingCallbacks & {
        nSubsteps: number,
    }): GpuFixedSimulationStepResult {
        const simulationTimestepS = this.selectedSimulationTimestepS;
        const safeNSubsteps = Number.isFinite(nSubsteps)
            ? Math.max(0, Math.floor(nSubsteps))
            : 0;

        if (this.destroyed) {
            return {
                nSimulationSubsteps: 0,
                simulationTimestepS,
                simulatedTimeS: 0,
            };
        }

        if (safeNSubsteps === 0) {
            this.renderStillFrame({
                onGpuTimeUpdate,
                onAnimationFrameTimeUpdate,
            });

            return {
                nSimulationSubsteps: 0,
                simulationTimestepS,
                simulatedTimeS: 0,
            };
        }

        const frameStartMs = performance.now();
        const commandEncoder = this.device.createCommandEncoder({
            label: "fixed simulation frame command encoder",
        });
        const enableInteraction = this.isInteracting();
        const shouldSampleParticleSpeed =
            safeNSubsteps > 0
            && this.particleSpeedReductionPipelineManager.canScheduleReadback();
        const measureGpuTimestamps = safeNSubsteps > 0 && this.canMeasureGpuTimestamps();
        this.performanceMeasurementManager?.setEnabled(measureGpuTimestamps);
        const {
            frameRenderUsesSsao,
            framePrerenderTimestampBaseIndex,
            timestampQueryCount,
        } = this.frameTimestampMetadata();

        this.writeSimulationTimingUniforms(simulationTimestepS);

        if (shouldSampleParticleSpeed) {
            this.particleSpeedReductionPipelineManager.reset({ commandEncoder });
        }

        this.addSimulationStepsComputePass({
            commandEncoder,
            nSimulationSteps: safeNSubsteps,
            measureGpuTimestamps,
            recordParticleSpeed: shouldSampleParticleSpeed,
            enableInteraction,
        });

        if (shouldSampleParticleSpeed) {
            this.particleSpeedReductionPipelineManager.copyToReadback({ commandEncoder });
        }

        this.addRender(commandEncoder, measureGpuTimestamps);

        if (measureGpuTimestamps && this.performanceMeasurementManager !== null) {
            this.performanceMeasurementManager.addResolve(commandEncoder, timestampQueryCount);
        }

        this.device.queue.submit([commandEncoder.finish()]);
        onAnimationFrameTimeUpdate?.(performance.now() - frameStartMs);

        if (shouldSampleParticleSpeed) {
            this.particleSpeedReductionPipelineManager.mapMaxSpeed(maxSpeed => {
                this.latestMaxParticleSpeed = maxSpeed;
            })
                .catch(error => {
                    console.error(error);
                });
        }

        if (measureGpuTimestamps) {
            this.mapSubmittedGpuTimes({
                nSimulationSubsteps: safeNSubsteps,
                frameRenderUsesSsao,
                framePrerenderTimestampBaseIndex,
                onGpuTimeUpdate,
            });
        }

        return {
            nSimulationSubsteps: safeNSubsteps,
            simulationTimestepS,
            simulatedTimeS: safeNSubsteps * simulationTimestepS,
        };
    }

    advanceFixedSimulationFrame(callbacks: GpuFrameTimingCallbacks = {}) {
        return this.advanceFixedSimulationSubsteps({
            nSubsteps: this.selectedSimulationSubstepsPerMaxStep,
            ...callbacks,
        });
    }

    updateColliderTransformMat(transformMat: Mat4) {
        const colliderTransformIsIdentity = mat4IsIdentity(transformMat);
        this.colliderTransformIsIdentity = colliderTransformIsIdentity;
        this.uniformsManager.writeColliderTransformMat(transformMat);
        this.uniformsManager.writeColliderTransformInv(mat4.inverse(transformMat));
        this.uniformsManager.writeColliderTransformIsIdentity(colliderTransformIsIdentity);
        this.writeColliderWorldBounds(transformMat);
    }

    updateColliderVel(transform: [number, number, number]) {
        const speedSquared =
            transform[0] * transform[0]
            + transform[1] * transform[1]
            + transform[2] * transform[2];
        const colliderVelocityIsZero = speedSquared <= COLLIDER_VELOCITY_ZERO_EPSILON_SQUARED;
        this.colliderVelocityIsZero = colliderVelocityIsZero;
        this.currentColliderSpeed = Math.sqrt(speedSquared);
        this.uniformsManager.writeColliderVel(transform);
        this.uniformsManager.writeColliderVelocityIsZero(colliderVelocityIsZero);
    }

    private writeColliderSdfMetadata(
        minCoords: [number, number, number],
        maxCoords: [number, number, number],
    ) {
        const extent: [number, number, number] = [
            maxCoords[0] - minCoords[0],
            maxCoords[1] - minCoords[1],
            maxCoords[2] - minCoords[2],
        ];
        const valid = extent.every(dimension => Number.isFinite(dimension) && dimension > 1e-6);
        const gridScale: [number, number, number] = valid
            ? [
                COLLIDER_SDF_LAST_COORD / extent[0],
                COLLIDER_SDF_LAST_COORD / extent[1],
                COLLIDER_SDF_LAST_COORD / extent[2],
            ]
            : [0, 0, 0];
        const cellSize: [number, number, number] = valid
            ? [
                extent[0] / COLLIDER_SDF_LAST_COORD,
                extent[1] / COLLIDER_SDF_LAST_COORD,
                extent[2] / COLLIDER_SDF_LAST_COORD,
            ]
            : [1, 1, 1];
        this.colliderSdfMaxCellSize = Math.max(...cellSize);
        this.colliderSdfValid = valid;

        this.uniformsManager.writeColliderSdfGridScale(gridScale);
        this.uniformsManager.writeColliderSdfCellSize(cellSize);
        this.uniformsManager.writeColliderSdfValid(valid);
        this.uniformsManager.writeColliderSdfMaxCellSize(this.colliderSdfMaxCellSize);
    }

    private writeColliderWorldBounds(transformMat: Mat4) {
        const margin = this.colliderSdfMaxCellSize * 2;
        const localMin = [
            this.colliderMinCoords[0] - margin,
            this.colliderMinCoords[1] - margin,
            this.colliderMinCoords[2] - margin,
        ];
        const localMax = [
            this.colliderMaxCoords[0] + margin,
            this.colliderMaxCoords[1] + margin,
            this.colliderMaxCoords[2] + margin,
        ];
        const worldMin: [number, number, number] = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
        const worldMax: [number, number, number] = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];

        for (const x of [localMin[0], localMax[0]]) {
            for (const y of [localMin[1], localMax[1]]) {
                for (const z of [localMin[2], localMax[2]]) {
                    const worldX = transformMat[0] * x + transformMat[4] * y + transformMat[8] * z + transformMat[12];
                    const worldY = transformMat[1] * x + transformMat[5] * y + transformMat[9] * z + transformMat[13];
                    const worldZ = transformMat[2] * x + transformMat[6] * y + transformMat[10] * z + transformMat[14];

                    worldMin[0] = Math.min(worldMin[0], worldX);
                    worldMin[1] = Math.min(worldMin[1], worldY);
                    worldMin[2] = Math.min(worldMin[2], worldZ);
                    worldMax[0] = Math.max(worldMax[0], worldX);
                    worldMax[1] = Math.max(worldMax[1], worldY);
                    worldMax[2] = Math.max(worldMax[2], worldZ);
                }
            }
        }

        this.uniformsManager.writeColliderWorldMinCoords(worldMin);
        this.uniformsManager.writeColliderWorldMaxCoords(worldMax);
    }

    private writeSimulationTimingUniforms(timestepS: number) {
        const safeTimestepS = Number.isFinite(timestepS)
            ? Math.max(timestepS, MIN_SIMULATION_TIMESTEP_S)
            : MIN_SIMULATION_TIMESTEP_S;

        const maxStableParticleSpeed = Math.min(
            CFL_NUMBER * this.minGridCellDim / safeTimestepS,
            MAX_FIXED_POINT_GRID_SPEED,
        );
        const maxStableParticleDisplacement = CFL_NUMBER * this.minGridCellDim;

        if (timestepS !== this.lastWrittenSimulationTimestepS) {
            this.uniformsManager.writeSimulationTimestepS(timestepS);
            this.lastWrittenSimulationTimestepS = timestepS;
        }

        if (maxStableParticleSpeed !== this.lastWrittenMaxStableParticleSpeed) {
            this.uniformsManager.writeMaxStableParticleSpeed(
                maxStableParticleSpeed,
                maxStableParticleDisplacement,
            );
            this.lastWrittenMaxStableParticleSpeed = maxStableParticleSpeed;
        }
    }

    private cflExternalAcceleration(isInteracting: boolean) {
        return GRAVITATIONAL_ACCELERATION_M_PER_S2
            + (isInteracting ? Math.abs(this.interactionStrength()) : 0);
    }

    private addSimulationStepsComputePass({
        commandEncoder,
        nSimulationSteps,
        measureGpuTimestamps,
        recordParticleSpeed,
        enableInteraction,
    }: {
        commandEncoder: GPUCommandEncoder,
        nSimulationSteps: number,
        measureGpuTimestamps: boolean,
        recordParticleSpeed: boolean,
        enableInteraction: boolean,
    }) {
        const simulationMethodType = this.getSimulationMethodType();
        const useStaticColliderPipeline = this.colliderTransformIsIdentity
            && this.colliderVelocityIsZero
            && this.colliderSdfValid;
        const useValidColliderPipeline = this.colliderSdfValid;

        if (simulationMethodType === GpuSimulationMethodType.FusedMlsMpm) {
            this.mpmPipelineManager.addFusedMlsMpmDispatchPassesBatch({
                commandEncoder,
                enableInteraction,
                nSimulationSteps,
                recordParticleSpeed,
                timestampWrites: this.performanceMeasurementManager !== null && measureGpuTimestamps
                    ? {
                        querySet: this.performanceMeasurementManager.querySet,
                        beginningOfPassWriteIndex: 0,
                        endOfPassWriteIndex: 1,
                    }
                    : undefined,
            });
            return;
        }

        const computePassEncoder = commandEncoder.beginComputePass({
            label: "simulation step compute pass",
            timestampWrites: this.performanceMeasurementManager !== null && measureGpuTimestamps
                ? {
                    querySet: this.performanceMeasurementManager.querySet,
                    beginningOfPassWriteIndex: 0,
                    endOfPassWriteIndex: 1,
                }
                : undefined,
        });

        this.mpmPipelineManager.bindCommonComputeGroups(computePassEncoder);

        switch (simulationMethodType) {
            case GpuSimulationMethodType.ExplicitMpm:
                this.mpmPipelineManager.addExplicitMpmDispatchesBatch({
                    computePassEncoder,
                    activeBlockDispatchBuffer: this.mpmManager.activeBlockDispatchBuffer,
                    enableInteraction,
                    nSimulationSteps,
                    recordParticleSpeed,
                    useStaticColliderPipeline,
                    useValidColliderPipeline,
                    bindCommonGroups: false,
                });
                break;

            case GpuSimulationMethodType.MlsMpm:
                this.mpmPipelineManager.addMlsMpmDispatchesBatch({
                    computePassEncoder,
                    activeBlockDispatchBuffer: this.mpmManager.activeBlockDispatchBuffer,
                    enableInteraction,
                    nSimulationSteps,
                    recordParticleSpeed,
                    useStaticColliderPipeline,
                    useValidColliderPipeline,
                    bindCommonGroups: false,
                });
                break;

        }

        computePassEncoder.end();
    }

    addRender(commandEncoder: GPUCommandEncoder, measureGpuTimestamps: boolean) {
        const renderMethod = this.renderMethod;
        if (renderMethod === null) return;

        this.uniformsManager.writeTime(Date.now());
        this.performanceMeasurementManager?.setPrerenderTimestampBaseIndex(this.renderUsesSsao ? 6 : 4);

        if (this.renderHasPrerenderPasses) {
            renderMethod.addPrerenderPasses(commandEncoder, this.depthTextureView);
        }

        const screenView = this.context.getCurrentTexture().createView();

        const renderPassEncoder = commandEncoder.beginRenderPass({
            label: "particles render pass",
            colorAttachments: [
                {
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: "clear",
                    storeOp: "store",
                    view: screenView,
                },
            ],
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
                depthClearValue: 1.0,
            },
            timestampWrites: this.performanceMeasurementManager !== null && measureGpuTimestamps
                ? {
                    querySet: this.performanceMeasurementManager.querySet,
                    beginningOfPassWriteIndex: 2,
                    endOfPassWriteIndex: 3,
                }
                : undefined,
        });


        this.rasterizeRenderPipelineManager?.addDraw(renderPassEncoder);
        //this.mpmGridRenderPipelineManager.addDraw(renderPassEncoder);
        this.environmentRenderPipelineManager.addDraw(renderPassEncoder);
        renderMethod.addCompositeDraw(renderPassEncoder);

        renderPassEncoder.end();

        if (this.renderUsesSsao) {
            const ssaoPassEncoder = commandEncoder.beginRenderPass({
                label: "ssao render pass",
                colorAttachments: [
                    {
                        view: screenView,
                        loadOp: "load",
                        storeOp: "store",
                    },
                ],
                timestampWrites: this.performanceMeasurementManager !== null && measureGpuTimestamps
                    ? {
                        querySet: this.performanceMeasurementManager.querySet,
                        beginningOfPassWriteIndex: 4,
                        endOfPassWriteIndex: 5,
                    }
                    : undefined,
            });

            this.ssaoPipelineManager.addDraw(ssaoPassEncoder);
            ssaoPassEncoder.end();
        }
    }

    loop({
        onGpuTimeUpdate,
        onAnimationFrameTimeUpdate,
        onUserControlUpdate,
    }: GpuFrameTimingCallbacks & {
        onUserControlUpdate?: () => void,
    } = {}) {
        if (this.destroyed) return () => {};

        let handle = 0;
        let canceled = false;

        let simulatedThroughTimeMs = performance.now();
        let gpuTimingFrameIndex = 0;
        let framesSinceLastParticleSpeedSample = PARTICLE_SPEED_SAMPLE_INTERVAL_FRAMES;
        let hasRequestedParticleSpeedSample = false;

        let lastFrameTime = 0;
        if (this.measurePerf) {
            lastFrameTime = performance.now();
        }

        const loop = () => {
            if (this.measurePerf) {
                const newFrameTime = performance.now();
                onAnimationFrameTimeUpdate?.(newFrameTime - lastFrameTime);
                lastFrameTime = newFrameTime;
            }

            onUserControlUpdate?.();

            const simulationMethodType = this.getSimulationMethodType();
            const canMeasureGpuTimestamps = this.performanceMeasurementManager !== null
                && this.renderMethod !== null
                && gpuTimingFrameIndex % GPU_TIMING_SAMPLE_INTERVAL_FRAMES === 0;
            gpuTimingFrameIndex++;

            const commandEncoder = this.device.createCommandEncoder({
                label: "loop command encoder",
            });

            const maxSimulationTimestepS = this.selectedMaxSimulationTimestepS;
            const simulationTimestepS = this.selectedSimulationTimestepS;
            const substepsPerMaxStep = this.selectedSimulationSubstepsPerMaxStep;
            const enableInteraction = this.isInteracting();

            this.writeSimulationTimingUniforms(simulationTimestepS);

            const nowMs = performance.now();
            const timeToSimulate = Math.max(0, nowMs - simulatedThroughTimeMs);
            const {
                nSubsteps,
                completedMaxSteps,
                shouldDropSimulationBacklog,
            } = calculateSimulationFrameSchedule({
                timeToSimulateMs: timeToSimulate,
                maxSimulationTimestepS,
                substepsPerMaxStep,
                oneSimulationStepPerFrame: this.oneSimulationStepPerFrame(),
                maxSimulationDriftMs: MAX_SIMULATION_DRIFT_MS,
                maxSimulationStepsPerFrame: MAX_SIMULATION_STEPS_PER_FRAME,
                maxSimulationSubstepsPerFrame: MAX_SIMULATION_SUBSTEPS_PER_FRAME,
            });

            const canSampleParticleSpeed = this.particleSpeedReductionPipelineManager.canScheduleReadback();
            const shouldForceParticleSpeedSample = !hasRequestedParticleSpeedSample
                || enableInteraction
                || this.currentColliderSpeed > MOVING_COLLIDER_SPEED_EPSILON;
            const particleSpeedSampleIntervalFrames = canRelaxParticleSpeedSampling({
                maxSimulationTimestepS,
                minGridCellDim: this.minGridCellDim,
                latestMaxParticleSpeed: this.latestMaxParticleSpeed,
                externalAcceleration: this.cflExternalAcceleration(enableInteraction),
                relaxedSampleIntervalFrames: PARTICLE_SPEED_RELAXED_SAMPLE_INTERVAL_FRAMES,
                speedHeadroom: PARTICLE_SPEED_RELAXED_CFL_HEADROOM,
                oneSimulationStepPerFrame: this.oneSimulationStepPerFrame(),
            })
                ? PARTICLE_SPEED_RELAXED_SAMPLE_INTERVAL_FRAMES
                : PARTICLE_SPEED_SAMPLE_INTERVAL_FRAMES;
            const speedSampleIsDue =
                framesSinceLastParticleSpeedSample >= particleSpeedSampleIntervalFrames;
            const shouldSampleParticleSpeed = nSubsteps > 0
                && (
                    shouldForceParticleSpeedSample
                    || speedSampleIsDue
                )
                && canSampleParticleSpeed;
            const measureGpuTimestamps = canMeasureGpuTimestamps
                && nSubsteps > 0
                && !shouldSampleParticleSpeed;
            this.performanceMeasurementManager?.setEnabled(measureGpuTimestamps);

            const frameRenderUsesSsao = this.renderUsesSsao;
            const framePrerenderTimestampBaseIndex = frameRenderUsesSsao ? 6 : 4;
            const framePrerenderPassCount = this.renderHasPrerenderPasses
                ? (this.prerenderPasses?.length ?? 0)
                : 0;
            const timestampQueryCount = framePrerenderTimestampBaseIndex + 2 * framePrerenderPassCount;

            if (shouldSampleParticleSpeed) {
                hasRequestedParticleSpeedSample = true;
                framesSinceLastParticleSpeedSample = 0;
            } else {
                framesSinceLastParticleSpeedSample++;
            }

            if (shouldDropSimulationBacklog) {
                simulatedThroughTimeMs = nowMs;
            }
            else {
                simulatedThroughTimeMs += completedMaxSteps * maxSimulationTimestepS * 1_000;
            }

            if (nSubsteps > 0) {
                if (shouldSampleParticleSpeed) {
                    this.particleSpeedReductionPipelineManager.reset({
                        commandEncoder,
                    });
                }

                this.addSimulationStepsComputePass({
                    commandEncoder,
                    nSimulationSteps: nSubsteps,
                    measureGpuTimestamps,
                    recordParticleSpeed: shouldSampleParticleSpeed,
                    enableInteraction,
                });

                if (shouldSampleParticleSpeed) {
                    this.particleSpeedReductionPipelineManager.copyToReadback({
                        commandEncoder,
                    });
                }
            }

            this.addRender(commandEncoder, measureGpuTimestamps);

            if (measureGpuTimestamps && this.performanceMeasurementManager !== null) {
                this.performanceMeasurementManager.addResolve(commandEncoder, timestampQueryCount);
            }

            this.device.queue.submit([commandEncoder.finish()]);

            if (shouldSampleParticleSpeed) {
                this.particleSpeedReductionPipelineManager.mapMaxSpeed(maxSpeed => {
                    this.latestMaxParticleSpeed = maxSpeed;
                })
                    .catch(error => {
                        console.error(error);
                    });
            }

            if (measureGpuTimestamps && this.performanceMeasurementManager !== null) {
                this.performanceMeasurementManager.mapTime(timestamps => {
                    const computeSimulationStepNs = timestamps[1] - timestamps[0];
                    const computeSimulationSubstepNs = nSubsteps > 0
                        ? computeSimulationStepNs / BigInt(nSubsteps)
                        : 0n;
                    const renderNs = timestamps[3] - timestamps[2];
                    const postprocessRenderNs = frameRenderUsesSsao
                        ? timestamps[5] - timestamps[4]
                        : 0n;
                    
                    if (this.#prerenderElapsedTimes !== null) {
                        for (let i = 0; i < this.#prerenderElapsedTimes.length; i++) {
                            const index = framePrerenderTimestampBaseIndex + 2 * i;
                            this.#prerenderElapsedTimes[i].elapsedTimeNs = timestamps[index + 1] - timestamps[index];
                        }
                    }
                    
                    onGpuTimeUpdate?.({
                        computeSimulationStepNs,
                        computeSimulationSubstepNs,
                        nSimulationSubsteps: nSubsteps,
                        renderNs,
                        postprocessRenderNs,
                    });
                })
                    .catch(error => {
                        console.error(error);
                        stop();
                    });
            }

            if (canceled) return;
            handle = requestAnimationFrame(loop);
        };

        handle = requestAnimationFrame(loop);

        const stop = () => {
            cancelAnimationFrame(handle);
            canceled = true;
        };

        return stop;
    }

    selectedMaxSimulationTimestepS = $derived.by(() => {
        switch (this.getSimulationMethodType()) {
            case GpuSimulationMethodType.ExplicitMpm:
                return this.explicitMpmMaxSimulationTimestepS();

            case GpuSimulationMethodType.MlsMpm:
            case GpuSimulationMethodType.FusedMlsMpm:
                return this.mlsMpmMaxSimulationTimestepS();
        }
    });

    private selectedCflLimitedSimulationTimestepS = $derived.by(() => {
        const maxSimulationTimestepS = this.selectedMaxSimulationTimestepS;
        const maxCflSpeed = Math.max(
            this.latestMaxParticleSpeed,
            this.currentColliderSpeed,
        );

        return calculateCflLimitedSimulationTimestepS({
            maxSimulationTimestepS,
            minGridCellDim: this.minGridCellDim,
            maxCflSpeed,
            externalAcceleration: this.cflExternalAcceleration(this.isInteracting()),
            elasticWaveSpeed: this.elasticWaveSpeed,
        });
    });

    selectedSimulationSubstepsPerMaxStep = $derived.by(() => {
        return calculateSimulationSubstepsPerMaxStep({
            maxSimulationTimestepS: this.selectedMaxSimulationTimestepS,
            cflLimitedSimulationTimestepS: this.selectedCflLimitedSimulationTimestepS,
        });
    });

    selectedSimulationTimestepS = $derived.by(() => {
        return calculateSimulationSubstepTimestepS({
            maxSimulationTimestepS: this.selectedMaxSimulationTimestepS,
            substepsPerMaxStep: this.selectedSimulationSubstepsPerMaxStep,
        });
    });

    async pickDepth(x: number, y: number): Promise<number | null> {
        if (this.destroyed || !this.depthTextureView) return null;
        return this.depthPicker.pick(this.depthTextureView, x, y);
    }
}
