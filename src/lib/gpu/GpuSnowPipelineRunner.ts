import type { Camera } from "$lib/Camera.svelte";
import { GpuPointsRenderPipelineManager } from "./pipelines/GpuPointsRenderPipelineManager";
import { GpuSimulationStepPipelineManager } from "./pipelines/GpuSimulationStepPipelineManager";
import { GpuUniformsBufferManager } from "./buffers/GpuUniformsBufferManager";
import { GpuMpmBufferManager } from "./buffers/GpuMpmBufferManager";
import { GpuRaymarchRenderPipelineManager } from "./pipelines/GpuRaymarchRenderPipelineManager";
import { GpuRenderMethodType, type GpuRenderMethod } from "./pipelines/GpuRenderMethod";
import { GpuPerformanceMeasurementBufferManager } from "./buffers/GpuPerformanceMeasurementBufferManager";
import { GpuMeshBufferManager } from "./buffers/GpuMeshBufferManager";
import { GpuColliderBufferManager } from "./buffers/GpuColliderBufferManager";
import { GpuParticleInitPipelineManager as GpuParticleScatterPipelineManager } from "./pipelines/GpuParticleScatterPipelineManager";
import { GpuRasterizeRenderPipelineManager } from "./pipelines/GpuRasterizeRenderPipelineManager";

const MAX_SIMULATION_DRIFT_MS = 1_000;
const FP_SCALE = 1024.0;

export class GpuSnowPipelineRunner {
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly nParticles: number;
    private readonly gridResolution: number;
    private readonly simulationTimestepS: number;
    private readonly camera: Camera;
    private depthTextureView: GPUTextureView;

    private readonly uniformsManager: GpuUniformsBufferManager;
    private readonly performanceMeasurementManager: GpuPerformanceMeasurementBufferManager | null;
    private readonly simulationStepPipelineManager: GpuSimulationStepPipelineManager;
    private readonly pointsRenderPipelineManager: GpuPointsRenderPipelineManager;
    private readonly raymarchRenderPipelineManager: GpuRaymarchRenderPipelineManager;
    private readonly rasterizeRenderPipelineManager: GpuRasterizeRenderPipelineManager;
    private readonly particleScatterPipelineManager: GpuParticleScatterPipelineManager;
    private readonly measurePerf: boolean;
    // debug
    // private readonly readbackBuffer : GPUBuffer;

    private readonly getRenderMethodType: () => GpuRenderMethodType;

    constructor({
        device,
        format,
        context,
        nParticles,
        gridResolution,
        simulationTimestepS,
        camera,
        meshVertices,
        colliderVertices,
        colliderIndices,
        getRenderMethodType,
        measurePerf,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        context: GPUCanvasContext,
        nParticles: number,
        gridResolution: number,
        simulationTimestepS: number,
        camera: Camera,
        meshVertices: number[][],
        colliderVertices: Float32Array;
        colliderIndices: Uint32Array;
        getRenderMethodType: () => GpuRenderMethodType,
        measurePerf: boolean,
    }) {
        this.device = device;
        this.context = context;
        this.nParticles = nParticles;
        this.gridResolution = gridResolution;
        this.simulationTimestepS = simulationTimestepS;

        this.camera = camera;

        const depthTexture = device.createTexture({
            size: [camera.screenDims.width(), camera.screenDims.height()],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.depthTextureView = depthTexture.createView();

        const uniformsManager = new GpuUniformsBufferManager({device});
        this.uniformsManager = uniformsManager;

        uniformsManager.writeSimulationTimestepS(simulationTimestepS);
        uniformsManager.writeGridResolution(gridResolution);
        uniformsManager.writeFixedPointScale(FP_SCALE);
        uniformsManager.writeGridMinCoords([-5, -5, 0]);
        uniformsManager.writeGridMaxCoords([5, 5, 4]);

        const mpmManager = new GpuMpmBufferManager({
            device,
            nParticles,
            gridResolution,
        });

        const meshManager = new GpuMeshBufferManager({device, vertices: meshVertices});
        uniformsManager.writeMeshMinCoords(meshManager.minCoords);
        uniformsManager.writeMeshMaxCoords(meshManager.maxCoords);

        const colliderManager = new GpuColliderBufferManager({
            device, 
            vertices: colliderVertices, 
            indices: colliderIndices
        });

        // debug
        // this.readbackBuffer = device.createBuffer({
        //         size: colliderManager.colliderIndicesBuffer.size,
        //         usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        // });
        
        // Compute
        const particleScatterPipelineManager = new GpuParticleScatterPipelineManager({
            device,
            particleDataBuffer: mpmManager.particleDataBuffer,
            meshVerticesBuffer: meshManager.meshVerticesBuffer,
            uniformsManager,
        });
        this.particleScatterPipelineManager = particleScatterPipelineManager;

        const simulationStepPipelineManager = new GpuSimulationStepPipelineManager({
            device,
            particleDataBuffer: mpmManager.particleDataBuffer,
            gridDataBuffer: mpmManager.gridDataBuffer,
            uniformsManager,
        });
        this.simulationStepPipelineManager = simulationStepPipelineManager;

        // Render
        const rasterizeRenderPipeline = new GpuRasterizeRenderPipelineManager({
            device, 
            format,
            depthFormat: "depth24plus",
            uniformsManager: uniformsManager,
            colliderManager: colliderManager
        });
        this.rasterizeRenderPipelineManager = rasterizeRenderPipeline;

        const pointsRenderPipelineManager = new GpuPointsRenderPipelineManager({device, format, uniformsManager, mpmManager});
        this.pointsRenderPipelineManager = pointsRenderPipelineManager;

        const raymarchRenderPipelineManager = new GpuRaymarchRenderPipelineManager({device, format, uniformsManager, mpmManager});
        this.raymarchRenderPipelineManager = raymarchRenderPipelineManager;

        this.getRenderMethodType = getRenderMethodType;

        this.performanceMeasurementManager = measurePerf
            ? new GpuPerformanceMeasurementBufferManager({device})
            : null;

        this.measurePerf = measurePerf;
    }

    resizeTextures(width: number, height: number) {
        const depthTexture = this.device.createTexture({
            size: [width, height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.depthTextureView = depthTexture.createView();
    }

    scatterParticlesInMeshVolume() {
        const commandEncoder = this.device.createCommandEncoder({
            label: "particle scatter command encoder",
        });

        this.particleScatterPipelineManager.addDispatch({
            commandEncoder,
            nParticles: this.nParticles,
        });

        this.device.queue.submit([commandEncoder.finish()]);
    }

    private async addSimulationStepsComputePass({
        commandEncoder,
        nSteps,
    }: {
        commandEncoder: GPUCommandEncoder,
        nSteps: number,
    }) {
        const computePassEncoder = commandEncoder.beginComputePass({
            label: "simulation step compute pass",
        });
        
        for (let i = 0; i < nSteps; i++) {
            this.simulationStepPipelineManager.addDispatch({
                computePassEncoder,
                numThreads: this.gridResolution ** 3,
                pipeline: this.simulationStepPipelineManager.gridClearComputePipeline,
            });
            
            this.simulationStepPipelineManager.addDispatch({
                computePassEncoder,
                numThreads: this.nParticles,
                pipeline: this.simulationStepPipelineManager.p2gComputePipeline,
            });

            this.simulationStepPipelineManager.addDispatch({
                computePassEncoder,
                numThreads: this.gridResolution ** 3,
                pipeline: this.simulationStepPipelineManager.gridComputePipeline,
            });

            this.simulationStepPipelineManager.addDispatch({
                computePassEncoder,
                numThreads: this.nParticles,
                pipeline: this.simulationStepPipelineManager.g2pComputePipeline,
            });
        }

        computePassEncoder.end();
    }

    async addRenderPass(commandEncoder: GPUCommandEncoder) {
        this.uniformsManager.writeViewProjMat(this.camera.viewProjMat);
        this.uniformsManager.writeViewProjInvMat(this.camera.viewProjInvMat);
        
        {
            const renderPassEncoder = commandEncoder.beginRenderPass({
                label: "particles render pass",
                colorAttachments: [
                    {
                        clearValue: { r: 0, g: 0, b: 0, a: 1 },
                        loadOp: "clear",
                        storeOp: "store",
                        view: this.context.getCurrentTexture().createView(),
                    },
                ],
                timestampWrites: this.performanceMeasurementManager !== null
                    ? {
                        querySet: this.performanceMeasurementManager.querySet,
                        beginningOfPassWriteIndex: 0,
                        endOfPassWriteIndex: 1,
                    }
                    : undefined,
            });

            this.selectRenderPipelineManager().addDraw(renderPassEncoder);
            renderPassEncoder.end();
        }

        {
            const renderPassEncoder = commandEncoder.beginRenderPass({
                label: "collider render pass",
                colorAttachments: [
                    {
                        clearValue: {
                            r: 0,
                            g: 0,
                            b: 0,
                            a: 1,
                        },

                        loadOp: "load",
                        storeOp: "store",
                        view: this.context.getCurrentTexture().createView(),
                    },
                ],
                depthStencilAttachment: {
                    view: this.depthTextureView,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                    depthClearValue: 1.0,
                },
            });
            
            this.rasterizeRenderPipelineManager.addDraw(renderPassEncoder);
            renderPassEncoder.end();
        }
    }

    loop({
        onGpuTimeUpdate,
        onAnimationFrameTimeUpdate,
    }: {
        onGpuTimeUpdate?: (ns: bigint) => void,
        onAnimationFrameTimeUpdate?: (ms: number) => void,
    } = {}) {
        let handle = 0;
        let canceled = false;

        const simulationTimestepMs = this.simulationTimestepS * 1_000;


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

            const commandEncoder = this.device.createCommandEncoder({
                label: "loop command encoder",
            });

            // debug
            // commandEncoder.copyBufferToBuffer(
            //     this.rasterizeRenderPipelineManager.colliderManager.colliderIndicesBuffer, 0,
            //     this.readbackBuffer, 0,
            //     this.rasterizeRenderPipelineManager.colliderManager.colliderIndicesBuffer.size
            // );
            // await this.readbackBuffer.mapAsync(GPUMapMode.READ);
            // const data = new Uint32Array(this.readbackBuffer.getMappedRange());
            // console.log(data);
            // this.readbackBuffer.unmap();

            // catch up the simulation to the current time
            let currentSimulationTime = simulationStartTime + nSimulationStep * simulationTimestepMs;
            let timeToSimulate = Date.now() - currentSimulationTime;

            const nSteps = Math.ceil(timeToSimulate / simulationTimestepMs);
            // if drifting too much, drop simulation steps 
            if (timeToSimulate <= MAX_SIMULATION_DRIFT_MS) {
                this.addSimulationStepsComputePass({
                    commandEncoder,
                    nSteps,
                });
            }
            nSimulationStep += nSteps;


            this.addRenderPass(commandEncoder);

            if (this.performanceMeasurementManager !== null) {
                this.performanceMeasurementManager.addResolve(commandEncoder);
            }

            this.device.queue.submit([commandEncoder.finish()]);
            // await this.device.queue.onSubmittedWorkDone();

            if (this.performanceMeasurementManager !== null) {
                this.performanceMeasurementManager.mapTime()
                    .then(elapsedTimeNs => {
                        if (elapsedTimeNs === null) return;
                        onGpuTimeUpdate?.(elapsedTimeNs);
                    });
            }

            if (canceled) return;
            handle = requestAnimationFrame(loop);
        };

        handle = requestAnimationFrame(loop);

        return () => {
            cancelAnimationFrame(handle);
            canceled = true;
        };
    }

    private selectRenderPipelineManager() {
        switch (this.getRenderMethodType()) {
            case GpuRenderMethodType.Points:
                return this.pointsRenderPipelineManager;
            
            case GpuRenderMethodType.Raymarch:
                return this.raymarchRenderPipelineManager;
        }
    }
}