import { mat4, type Mat4 } from "wgpu-matrix";
import type { Camera } from "$lib/components/simulationViewer/Camera.svelte";
import { GpuPointsRenderPipelineManager } from "./pointsRender/GpuPointsRenderPipelineManager";
import { GpuMpmPipelineManager } from "./mpm/GpuMpmPipelineManager";
import { GpuUniformsBufferManager } from "./uniforms/GpuUniformsBufferManager";
import { GpuMpmBufferManager } from "./mpm/GpuMpmBufferManager";
import { GpuRenderMethodType, type GpuRenderMethod } from "./GpuRenderMethod";
import { GpuPerformanceMeasurementBufferManager } from "./performanceMeasurement/GpuPerformanceMeasurementBufferManager";
import { GpuMeshBufferManager } from "./particleInitialize/GpuMeshBufferManager";
import { GpuColliderBufferManager } from "./collider/GpuColliderBufferManager";
import { GpuParticleInitializePipelineManager } from "./particleInitialize/GpuParticleInitializePipelineManager";
import { GpuRasterizeRenderPipelineManager } from "./collider/GpuRasterizeRenderPipelineManager";
import { GpuMpmGridRenderPipelineManager } from "./mpmGridRender/GpuMpmGridRenderPipelineMager";
import { GpuVolumetricBufferManager } from "./volumetric/GpuVolumetricBufferManager";
import { GpuVolumetricRenderPipelineManager } from "./volumetric/GpuVolumetricRenderPipelineManager";
import { GpuSsfrRenderPipelineManager } from "./ssfr/GpuSsfrRenderPipelineManager";
import { GpuMarchingCubesRenderPipelineManager } from "./marchingCubes/GpuMarchingCubesRenderPipelineManager";
import type { ColliderGeometry } from "./collider/GpuColliderBufferManager";
import { GpuSimulationMethodType } from "./GpuSimulationMethod";
import { GpuEnvironmentRenderPipelineManager } from "./environmentMap/GpuEnvironmentRenderPipelineManager";
import { GpuEnvironmentTextureManager } from "./environmentMap/GpuEnvironmentTextureManager";
import { untrack } from "svelte";
import { PrerenderPassElapsedTime } from "$lib/components/simulationViewer/PrerenderPassElapsedTime.svelte";
import { GpuSsaoPipelineManager } from "./ssao/GpuSsaoPipelineManager";
import { GpuDepthPicker } from "./GpuDepthPicker";

const MAX_SIMULATION_DRIFT_MS = 250;
const FP_SCALE = 65536;

export class GpuSnowPipelineRunner {
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly nParticles: number;
    private readonly explicitMpmSimulationTimestepS: () => number;
    private readonly pbmpmSimulationTimestepS: () => number;
    private readonly camera: Camera;
    private depthTextureView: GPUTextureView;

    readonly uniformsManager: GpuUniformsBufferManager;
    private readonly performanceMeasurementManager: GpuPerformanceMeasurementBufferManager | null;
    private readonly mpmPipelineManager: GpuMpmPipelineManager;
    private readonly rasterizeRenderPipelineManager: GpuRasterizeRenderPipelineManager;
    private readonly mpmGridRenderPipelineManager: GpuMpmGridRenderPipelineManager;
    private readonly particleInitializePipelineManager: GpuParticleInitializePipelineManager;
    private readonly environmentRenderPipelineManager: GpuEnvironmentRenderPipelineManager;
    private readonly ssaoPipelineManager: GpuSsaoPipelineManager;
    private readonly depthPicker: GpuDepthPicker;

    private depthTexture: GPUTexture | null = null;

    private renderMethod = $state<GpuRenderMethod | null>(null);
    readonly prerenderPasses = $derived(this.renderMethod?.prerenderPasses() ?? null);
    #prerenderElapsedTimes = $state<PrerenderPassElapsedTime[] | null>(null);
    readonly prerenderElapsedTimes = $derived(this.#prerenderElapsedTimes);


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
        explicitMpmSimulationTimestepS,
        pbmpmSimulationTimestepS,
        camera,
        meshVertices,
        collider,
        getSimulationMethodType,
        getRenderMethodType,
        oneSimulationStepPerFrame,
        environmentImageBitmap,
        measurePerf,
        width,
        height,
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
        explicitMpmSimulationTimestepS: () => number,
        pbmpmSimulationTimestepS: () => number,
        camera: Camera,
        meshVertices: number[][],
        collider: ColliderGeometry,
        getSimulationMethodType: () => GpuSimulationMethodType,
        getRenderMethodType: () => GpuRenderMethodType,
        oneSimulationStepPerFrame: () => boolean,
        environmentImageBitmap: ImageBitmap,
        measurePerf: boolean,
        width: () => number,
        height: () => number,
    }) {
        this.device = device;
        this.context = context;
        this.nParticles = nParticles;
        this.explicitMpmSimulationTimestepS = explicitMpmSimulationTimestepS;
        this.pbmpmSimulationTimestepS = pbmpmSimulationTimestepS;

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

        uniformsManager.writeGridResolution([gridResolutionX, gridResolutionY, gridResolutionZ]);
        uniformsManager.writeGridCellDims([
            (gridMaxCoords[0] - gridMinCoords[0]) / gridResolutionX,
            (gridMaxCoords[1] - gridMinCoords[1]) / gridResolutionY,
            (gridMaxCoords[2] - gridMinCoords[2]) / gridResolutionZ,
        ]);
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
            gridResolutionX,
            gridResolutionY,
            gridResolutionZ,
        });
        this.mpmManager = mpmManager;

        const meshManager = new GpuMeshBufferManager({device, vertices: meshVertices});
        uniformsManager.writeMeshMinCoords(meshManager.minCoords);
        uniformsManager.writeMeshMaxCoords(meshManager.maxCoords);

        const colliderManager = new GpuColliderBufferManager({
            device, 
            vertices: collider.positions, 
            normals: collider.normals, 
            uvs: collider.uvs,
            materialIndices: collider.materialIndices,
            textures: collider.textures,
            indices: collider.indices,
        });
        uniformsManager.writeColliderObjects(collider.objects);
        uniformsManager.writeColliderNumObjects(Math.min(collider.objects.length, 1024));
        uniformsManager.writeColliderMinCoords(colliderManager.minCoords);
        uniformsManager.writeColliderMaxCoords(colliderManager.maxCoords);
        uniformsManager.writeColliderTransformMat(mat4.identity());
        uniformsManager.writeColliderTransformInv(mat4.identity());
        uniformsManager.writeColliderVel([0.0, 0.0, 0.0]);

        // Compute
        const particleInitializePipelineManager = new GpuParticleInitializePipelineManager({
            device,
            particleDataBuffer: mpmManager.particleDataBuffer,
            meshVerticesBuffer: meshManager.meshVerticesBuffer,
            uniformsManager,
        });
        this.particleInitializePipelineManager = particleInitializePipelineManager;


        const mpmPipelineManager = new GpuMpmPipelineManager({
            device,
            uniformsManager,
            mpmManager,
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
            $effect(() => this.uniformsManager.writeSimulationTimestepS(this.selectedSimulationTimestepS));


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
                }
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
        this.uniformsManager.writeSimulationTimestepS(this.selectedSimulationTimestepS);
        
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
    }

    updateColliderVel(transform: [number, number, number]) {
        this.uniformsManager.writeColliderVel(transform);
    }

    private async addSimulationStepsComputePass({
        commandEncoder,
        nSimulationSteps,
    }: {
        commandEncoder: GPUCommandEncoder,
        nSimulationSteps: number,
    }) {
        const simulationMethodType = this.getSimulationMethodType();

        switch (simulationMethodType) {
            case GpuSimulationMethodType.ExplicitMpm:
                this.uniformsManager.writeUsePbmpm(false);
                break;
                
            case GpuSimulationMethodType.Pbmpm:
                this.uniformsManager.writeUsePbmpm(true);
                break;
        }

        const computePassEncoder = commandEncoder.beginComputePass({
            label: "simulation step compute pass",
            timestampWrites: this.performanceMeasurementManager !== null
                ? {
                    querySet: this.performanceMeasurementManager.querySet,
                    beginningOfPassWriteIndex: 0,
                    endOfPassWriteIndex: 1,
                }
                : undefined,
        });

        for (let i = 0; i < nSimulationSteps; i++) {
            switch (simulationMethodType) {
                case GpuSimulationMethodType.ExplicitMpm:
                    this.mpmPipelineManager.addExplicitMpmDispatches({
                        computePassEncoder,
                        nParticles: this.mpmManager.nParticles,
                    });
                    break;
                    
                case GpuSimulationMethodType.Pbmpm:
                    this.mpmPipelineManager.addPbmpmDispatches({
                        computePassEncoder,
                        nParticles: this.mpmManager.nParticles,
                    });
                    break;
            }
        }

        computePassEncoder.end();
    }

    async addRender(commandEncoder: GPUCommandEncoder) {
        if (this.renderMethod === null) return;

        this.uniformsManager.writeTime(Date.now());

        if (this.renderMethod.prerenderPasses().length > 0) {
            this.renderMethod.addPrerenderPasses(commandEncoder, this.depthTextureView);
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
            timestampWrites: this.performanceMeasurementManager !== null
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
        this.renderMethod.addCompositeDraw(renderPassEncoder);

        renderPassEncoder.end();

        const ssaoPassEncoder = commandEncoder.beginRenderPass({
            label: "ssao render pass",
            colorAttachments: [
                {
                    view: screenView,
                    loadOp: "load",
                    storeOp: "store",
                },
            ],
            timestampWrites: this.performanceMeasurementManager !== null
                ? {
                    querySet: this.performanceMeasurementManager.querySet,
                    beginningOfPassWriteIndex: 4,
                    endOfPassWriteIndex: 5,
                }
                : undefined,
        });

        if ([GpuRenderMethodType.MarchingCubes, GpuRenderMethodType.Points].includes(this.getRenderMethodType())) {
            this.ssaoPipelineManager.addDraw(ssaoPassEncoder);
        }
        ssaoPassEncoder.end();
    }

    loop({
        onGpuTimeUpdate,
        onAnimationFrameTimeUpdate,
        onUserControlUpdate,
    }: {
        onGpuTimeUpdate?: (times: {
            computeSimulationStepNs: bigint,
            renderNs: bigint,
            postprocessRenderNs: bigint,
        }) => void,
        onAnimationFrameTimeUpdate?: (ms: number) => void,
        onUserControlUpdate?: () => void,
    } = {}) {
        let handle = 0;
        let canceled = false;

        const simulationTimestepMs = this.selectedSimulationTimestepS * 1_000;


        let nSimulationStep = 0;
        let simulationStartTime = Date.now();
        
        let lastFrameTime = 0;
        if (this.measurePerf) {
            lastFrameTime = performance.now();
        }

        const loop = async () => {
            if (this.measurePerf) {
                const newFrameTime = performance.now();
                onAnimationFrameTimeUpdate?.(newFrameTime - lastFrameTime);
                lastFrameTime = newFrameTime;
            }

            onUserControlUpdate?.();

            const commandEncoder = this.device.createCommandEncoder({
                label: "loop command encoder",
            });

            // catch up the simulation to the current time
            let currentSimulationTime = simulationStartTime + nSimulationStep * simulationTimestepMs;
            let timeToSimulate = Date.now() - currentSimulationTime;

            let nSteps = Math.ceil(timeToSimulate / simulationTimestepMs);
            nSimulationStep += nSteps;
            if (this.oneSimulationStepPerFrame()) {
                nSteps = Math.min(1, nSteps);

                this.addSimulationStepsComputePass({
                    commandEncoder,
                    nSimulationSteps: nSteps,
                });
            }
            else {
                // if drifting too much, drop simulation steps 
                if (timeToSimulate <= MAX_SIMULATION_DRIFT_MS) {
                    this.addSimulationStepsComputePass({
                        commandEncoder,
                        nSimulationSteps: nSteps,
                    });
                }
            }
            

            this.addRender(commandEncoder);

            if (this.performanceMeasurementManager !== null) {
                this.performanceMeasurementManager.addResolve(commandEncoder);
            }

            this.device.queue.submit([commandEncoder.finish()]);

            if (this.performanceMeasurementManager !== null) {
                this.performanceMeasurementManager.mapTime(timestamps => {
                    const computeSimulationStepNs = timestamps[1] - timestamps[0];
                    const renderNs = timestamps[3] - timestamps[2];
                    const postprocessRenderNs = timestamps[5] - timestamps[4];
                    
                    if (this.#prerenderElapsedTimes !== null) {
                        for (let i = 0; i < this.#prerenderElapsedTimes.length; i++) {
                            const index = 2 * (i + 3);
                            this.#prerenderElapsedTimes[i].elapsedTimeNs = timestamps[index + 1] - timestamps[index];
                        }
                    }
                    
                    onGpuTimeUpdate?.({
                        computeSimulationStepNs,
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

    selectedSimulationTimestepS = $derived.by(() => {
        switch (this.getSimulationMethodType()) {
            case GpuSimulationMethodType.ExplicitMpm:
                return this.explicitMpmSimulationTimestepS();

            case GpuSimulationMethodType.Pbmpm:
                return this.pbmpmSimulationTimestepS();
        }
    });

    async pickDepth(x: number, y: number): Promise<number | null> {
        if (!this.depthTextureView) return null;
        return this.depthPicker.pick(this.depthTextureView, x, y);
    }
}
