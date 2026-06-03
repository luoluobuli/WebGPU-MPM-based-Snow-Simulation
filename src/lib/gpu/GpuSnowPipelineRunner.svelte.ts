import { mat4, type Mat4 } from "wgpu-matrix";
import type { Camera } from "$lib/components/simulationViewer/Camera.svelte";
import { GpuPointsRenderPipelineManager } from "./pointsRender/GpuPointsRenderPipelineManager";
import { GpuMpmPipelineManager } from "./mpm/GpuMpmPipelineManager";
import { GpuUniformsBufferManager } from "./uniforms/GpuUniformsBufferManager";
import { GpuMpmBufferManager } from "./mpm/GpuMpmBufferManager";
import { GpuRenderMethodType, type GpuRenderMethod } from "./GpuRenderMethod";
import { GpuPerformanceMeasurementBufferManager } from "./performanceMeasurement/GpuPerformanceMeasurementBufferManager";
import { GpuMeshBufferManager } from "./particleInitialize/GpuMeshBufferManager";
import { GpuSpawnVolumeBufferManager } from "./particleInitialize/GpuSpawnVolumeBufferManager";
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
import {
    CFL_NUMBER,
    calculateCflLimitedSimulationTimestepS,
    calculateSimulationSubstepTimestepS,
    calculateSimulationSubstepsPerMaxStep,
    MIN_SIMULATION_TIMESTEP_S,
} from "./simulationTimestep";

const MAX_SIMULATION_DRIFT_MS = 250;
const MAX_SIMULATION_STEPS_PER_FRAME = 128;
const MAX_SIMULATION_SUBSTEPS_PER_FRAME = 512;
const GRAVITATIONAL_ACCELERATION_M_PER_S2 = 9.81;
const FP_SCALE = 65536;
const GPU_TIMING_SAMPLE_INTERVAL_FRAMES = 8;
const PARTICLE_SPEED_SAMPLE_INTERVAL_FRAMES = 4;
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

type SpawnMeshObject = {
    min: [number, number, number];
    max: [number, number, number];
    startVertex: number;
    countVertices: number;
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
    private readonly colliderMinCoords: [number, number, number];
    private readonly colliderMaxCoords: [number, number, number];
    private colliderSdfMaxCellSize = 1;
    private depthTextureView: GPUTextureView;

    readonly uniformsManager: GpuUniformsBufferManager;
    private readonly performanceMeasurementManager: GpuPerformanceMeasurementBufferManager | null;
    private readonly mpmPipelineManager: GpuMpmPipelineManager;
    private readonly particleSpeedReductionPipelineManager: GpuParticleSpeedReductionPipelineManager;
    private readonly rasterizeRenderPipelineManager: GpuRasterizeRenderPipelineManager;
    private readonly mpmGridRenderPipelineManager: GpuMpmGridRenderPipelineManager;
    private readonly particleInitializePipelineManager: GpuParticleInitializePipelineManager;
    private readonly environmentRenderPipelineManager: GpuEnvironmentRenderPipelineManager;
    private readonly ssaoPipelineManager: GpuSsaoPipelineManager;
    private readonly depthPicker: GpuDepthPicker;

    private depthTexture: GPUTexture | null = null;

    private renderMethod = $state<GpuRenderMethod | null>(null);
    readonly prerenderPasses = $derived(this.renderMethod?.prerenderPasses() ?? null);
    private renderHasPrerenderPasses = false;
    private renderUsesSsao = false;
    #prerenderElapsedTimes = $state<PrerenderPassElapsedTime[] | null>(null);
    readonly prerenderElapsedTimes = $derived(this.#prerenderElapsedTimes);
    private latestMaxParticleSpeed = $state(0);
    private currentColliderSpeed = $state(0);
    private lastWrittenSimulationTimestepS = Number.NaN;
    private lastWrittenMaxStableParticleSpeed = Number.NaN;


    private readonly mpmManager: GpuMpmBufferManager;

    private readonly measurePerf: boolean;
    // debug
    // private readonly readbackBuffer : GPUBuffer;
    // v : [number, number, number] = [0.0, 0.0, 0.0];

    private readonly getSimulationMethodType: () => GpuSimulationMethodType;
    private readonly getRenderMethodType: () => GpuRenderMethodType;
    private readonly oneSimulationStepPerFrame: () => boolean;

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
        meshVertices,
        meshObjects,
        collider,
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
        meshVertices: number[][],
        meshObjects?: SpawnMeshObject[],
        collider: ColliderGeometry,
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

        this.camera = camera;

        const depthTexture = device.createTexture({
            size: [camera.screenDims.width(), camera.screenDims.height()],
            format: "depth32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.depthTextureView = depthTexture.createView();

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

        const meshManager = new GpuMeshBufferManager({device, vertices: meshVertices});
        uniformsManager.writeMeshMinCoords(meshManager.minCoords);
        uniformsManager.writeMeshMaxCoords(meshManager.maxCoords);

        const spawnVolumeManager = new GpuSpawnVolumeBufferManager({
            device,
            vertices: meshVertices,
            nParticles,
            objects: meshObjects,
        });

        const colliderManager = new GpuColliderBufferManager({
            device, 
            vertices: collider.positions, 
            normals: collider.normals, 
            uvs: collider.uvs,
            materialIndices: collider.materialIndices,
            textures: collider.textures,
            indices: collider.indices,
        });
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
            sparseGridBuffer: mpmManager.sparseGridBuffer,
            gridMassBuffer: mpmManager.gridMassBuffer,
            gridMomentumXBuffer: mpmManager.gridMomentumXBuffer,
            gridMomentumYBuffer: mpmManager.gridMomentumYBuffer,
            gridMomentumZBuffer: mpmManager.gridMomentumZBuffer,
            maxParticleSpeedBuffer: this.particleSpeedReductionPipelineManager.maxSpeedBuffer,
            activeBlockDispatchBuffer: mpmManager.activeBlockDispatchBuffer,
            uniformsManager,
            colliderManager,
        });

        this.mpmPipelineManager = mpmPipelineManager;

        // Render
        const rasterizeRenderPipeline = new GpuRasterizeRenderPipelineManager({
            device, 
            format,
            depthFormat: "depth32float",
            uniformsManager: uniformsManager,
            colliderManager: colliderManager
        });
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

        $effect.root(() => {
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
                const renderMethodType = getRenderMethodType();
                if (renderMethodType === lastRenderMethodType) return;

                this.renderMethod?.destroy();
                lastRenderMethodType = renderMethodType;




                switch (renderMethodType) {
                    case GpuRenderMethodType.Points:
                        this.renderMethod = new GpuPointsRenderPipelineManager({
                            device,
                            format,
                            depthFormat: "depth32float",
                            uniformsManager,
                            mpmManager,
                        });
                        break;

                    case GpuRenderMethodType.Splats:
                        this.renderMethod = new GpuSplatsRenderPipelineManager({
                            device,
                            format,
                            depthFormat: "depth32float",
                            uniformsManager,
                            mpmManager,
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
            };
        });
    }

    scatterParticlesInMeshVolume() {
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

    updateColliderTransformMat(transformMat: Mat4) {
        this.uniformsManager.writeColliderTransformMat(transformMat);
        this.uniformsManager.writeColliderTransformInv(mat4.inverse(transformMat));
        this.uniformsManager.writeColliderTransformIsIdentity(mat4IsIdentity(transformMat));
        this.writeColliderWorldBounds(transformMat);
    }

    updateColliderVel(transform: [number, number, number]) {
        const speedSquared =
            transform[0] * transform[0]
            + transform[1] * transform[1]
            + transform[2] * transform[2];
        this.currentColliderSpeed = Math.sqrt(speedSquared);
        this.uniformsManager.writeColliderVel(transform);
        this.uniformsManager.writeColliderVelocityIsZero(speedSquared <= COLLIDER_VELOCITY_ZERO_EPSILON_SQUARED);
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

        const maxStableParticleSpeed = CFL_NUMBER * this.minGridCellDim / safeTimestepS;
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


        this.rasterizeRenderPipelineManager.addDraw(renderPassEncoder);
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
    }: {
        onGpuTimeUpdate?: (times: {
            computeSimulationStepNs: bigint,
            computeSimulationSubstepNs: bigint,
            nSimulationSubsteps: number,
            renderNs: bigint,
            postprocessRenderNs: bigint,
        }) => void,
        onAnimationFrameTimeUpdate?: (ms: number) => void,
        onUserControlUpdate?: () => void,
    } = {}) {
        let handle = 0;
        let canceled = false;

        let simulatedThroughTimeMs = performance.now();
        let gpuTimingFrameIndex = 0;
        let particleSpeedSampleFrameIndex = 0;
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

            const maxSimulationTimestepMs = maxSimulationTimestepS * 1_000;
            const nowMs = performance.now();
            const timeToSimulate = Math.max(0, nowMs - simulatedThroughTimeMs);

            let nSteps = 0;
            let shouldDropSimulationBacklog = false;
            if (this.oneSimulationStepPerFrame()) {
                nSteps = timeToSimulate > 0 ? 1 : 0;
                shouldDropSimulationBacklog = true;
            }
            else {
                if (timeToSimulate <= MAX_SIMULATION_DRIFT_MS) {
                    nSteps = Math.min(
                        Math.ceil(timeToSimulate / maxSimulationTimestepMs),
                        MAX_SIMULATION_STEPS_PER_FRAME,
                    );
                }
                else {
                    shouldDropSimulationBacklog = true;
                }
            }

            const nSubsteps = Math.min(
                nSteps * substepsPerMaxStep,
                MAX_SIMULATION_SUBSTEPS_PER_FRAME,
            );
            const measureGpuTimestamps = canMeasureGpuTimestamps && nSubsteps > 0;
            this.performanceMeasurementManager?.setEnabled(measureGpuTimestamps);

            const frameRenderUsesSsao = this.renderUsesSsao;
            const framePrerenderTimestampBaseIndex = frameRenderUsesSsao ? 6 : 4;
            const framePrerenderPassCount = this.renderHasPrerenderPasses
                ? (this.prerenderPasses?.length ?? 0)
                : 0;
            const timestampQueryCount = framePrerenderTimestampBaseIndex + 2 * framePrerenderPassCount;

            const canSampleParticleSpeed = this.particleSpeedReductionPipelineManager.canScheduleReadback();
            const shouldSampleParticleSpeed = nSubsteps > 0 && (
                !hasRequestedParticleSpeedSample
                || enableInteraction
                || this.currentColliderSpeed > MOVING_COLLIDER_SPEED_EPSILON
                || particleSpeedSampleFrameIndex % PARTICLE_SPEED_SAMPLE_INTERVAL_FRAMES === 0
            ) && canSampleParticleSpeed;
            particleSpeedSampleFrameIndex++;
            if (shouldSampleParticleSpeed) {
                hasRequestedParticleSpeedSample = true;
            }

            const completedMaxSteps = substepsPerMaxStep > 0
                ? nSubsteps / substepsPerMaxStep
                : 0;

            if (shouldDropSimulationBacklog) {
                simulatedThroughTimeMs = nowMs;
            }
            else {
                simulatedThroughTimeMs += completedMaxSteps * maxSimulationTimestepMs;
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
                return this.mlsMpmMaxSimulationTimestepS();
        }
    });

    private selectedCflLimitedSimulationTimestepS = $derived.by(() => {
        const maxSimulationTimestepS = this.selectedMaxSimulationTimestepS;
        const maxCflSpeed = Math.max(
            this.latestMaxParticleSpeed,
            this.currentColliderSpeed,
        );
        const externalAcceleration = GRAVITATIONAL_ACCELERATION_M_PER_S2
            + (this.isInteracting() ? Math.abs(this.interactionStrength()) : 0);

        return calculateCflLimitedSimulationTimestepS({
            maxSimulationTimestepS,
            minGridCellDim: this.minGridCellDim,
            maxCflSpeed,
            externalAcceleration,
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
        if (!this.depthTextureView) return null;
        return this.depthPicker.pick(this.depthTextureView, x, y);
    }
}
