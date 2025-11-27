import { mat4, type Mat4 } from "wgpu-matrix";
import type { Camera } from "$lib/components/simulationViewer/Camera.svelte";
import { GpuPointsRenderPipelineManager } from "./pointsRender/GpuPointsRenderPipelineManager";
import { GpuMpmPipelineManager } from "./mpm/GpuMpmPipelineManager";
import { GpuUniformsBufferManager } from "./uniforms/GpuUniformsBufferManager";
import { GpuMpmBufferManager } from "./mpm/GpuMpmBufferManager";
import { GpuRaymarchRenderPipelineManager } from "./raymarchRender/GpuRaymarchRenderPipelineManager";
import { GpuRenderMethodType } from "./GpuRenderMethod";
import { GpuPerformanceMeasurementBufferManager } from "./performanceMeasurement/GpuPerformanceMeasurementBufferManager";
import { GpuMeshBufferManager } from "./particleInitialize/GpuMeshBufferManager";
import { GpuColliderBufferManager } from "./collider/GpuColliderBufferManager";
import { GpuParticleInitializePipelineManager } from "./particleInitialize/GpuParticleInitializePipelineManager";
import { GpuRasterizeRenderPipelineManager } from "./collider/GpuRasterizeRenderPipelineManager";
import { GpuMpmGridRenderPipelineManager } from "./mpmGridRender/GpuMpmGridRenderPipelineMager";
import { GpuVolumetricBufferManager } from "./volumetric/GpuVolumetricBufferManager";
import { GpuVolumetricRenderPipelineManager } from "./volumetric/GpuVolumetricRenderPipelineManager";
import type { ColliderGeometry } from "./collider/GpuColliderBufferManager";

const MAX_SIMULATION_DRIFT_MS = 1_000;
const FP_SCALE = 1024.0;

export class GpuSnowPipelineRunner {
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly nParticles: number;
    private readonly gridResolutionX: number;
    private readonly gridResolutionY: number;
    private readonly gridResolutionZ: number;
    private readonly simulationTimestepS: number;
    private readonly camera: Camera;
    private depthTextureView: GPUTextureView;

    private readonly uniformsManager: GpuUniformsBufferManager;
    private readonly performanceMeasurementManager: GpuPerformanceMeasurementBufferManager | null;
    private readonly mpmPipelineManager: GpuMpmPipelineManager;
    private readonly pointsRenderPipelineManager: GpuPointsRenderPipelineManager;
    private readonly raymarchRenderPipelineManager: GpuRaymarchRenderPipelineManager;
    private readonly rasterizeRenderPipelineManager: GpuRasterizeRenderPipelineManager;
    private readonly mpmGridRenderPipelineManager: GpuMpmGridRenderPipelineManager;
    private readonly volumetricBufferManager: GpuVolumetricBufferManager;
    private readonly volumetricRenderPipelineManager: GpuVolumetricRenderPipelineManager;
    private readonly particleInitializePipelineManager: GpuParticleInitializePipelineManager;

    private readonly measurePerf: boolean;
    // debug
    // private readonly readbackBuffer : GPUBuffer;
    // v : [number, number, number] = [0.0, 0.0, 0.0];

    private readonly getRenderMethodType: () => GpuRenderMethodType;

    constructor({
        device,
        format,
        context,
        nParticles,
        gridResolutionX,
        gridResolutionY,
        gridResolutionZ,
        simulationTimestepS,
        camera,
        meshVertices,
        collider,
        getRenderMethodType,
        measurePerf,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        context: GPUCanvasContext,
        nParticles: number,
        gridResolutionX: number,
        gridResolutionY: number,
        gridResolutionZ: number,
        simulationTimestepS: number,
        camera: Camera,
        meshVertices: number[][],
        collider: ColliderGeometry,
        getRenderMethodType: () => GpuRenderMethodType,
        measurePerf: boolean,
    }) {
        this.device = device;
        this.context = context;
        this.nParticles = nParticles;
        this.gridResolutionX = gridResolutionX;
        this.gridResolutionY = gridResolutionY;
        this.gridResolutionZ = gridResolutionZ;
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
        uniformsManager.writeGridResolution([gridResolutionX, gridResolutionY, gridResolutionZ]);
        uniformsManager.writeFixedPointScale(FP_SCALE);
        uniformsManager.writeGridMinCoords([-5, -5, 0]);
        uniformsManager.writeGridMaxCoords([5, 5, 4]);

        const mpmManager = new GpuMpmBufferManager({
            device,
            nParticles,
            gridResolutionX,
            gridResolutionY,
            gridResolutionZ,
        });

        const meshManager = new GpuMeshBufferManager({device, vertices: meshVertices});
        uniformsManager.writeMeshMinCoords(meshManager.minCoords);
        uniformsManager.writeMeshMaxCoords(meshManager.maxCoords);

        const colliderManager = new GpuColliderBufferManager({
            device, 
            vertices: collider.positions, 
            normals: collider.normals,
            indices: collider.indices,
        });
        uniformsManager.writeColliderMinCoords(colliderManager.minCoords);
        uniformsManager.writeColliderMaxCoords(colliderManager.maxCoords);
        uniformsManager.writeColliderTransformMat(mat4.identity());
        uniformsManager.writeColliderVel([0.0, 0.0, 0.0]);

        // debug
        // this.readbackBuffer = device.createBuffer({
        //         size: colliderManager.indicesBuffer.size,
        //         usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        // });
        
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
            particleDataBuffer: mpmManager.particleDataBuffer,
            gridDataBuffer: mpmManager.gridDataBuffer,
            uniformsManager,
        });
        this.mpmPipelineManager = mpmPipelineManager;

        // Render
        const rasterizeRenderPipeline = new GpuRasterizeRenderPipelineManager({
            device, 
            format,
            depthFormat: "depth24plus",
            uniformsManager: uniformsManager,
            colliderManager: colliderManager
        });
        this.rasterizeRenderPipelineManager = rasterizeRenderPipeline;

        const pointsRenderPipelineManager = new GpuPointsRenderPipelineManager({
            device,
            format,
            depthFormat: "depth24plus",
            uniformsManager,
            mpmManager,
        });
        this.pointsRenderPipelineManager = pointsRenderPipelineManager;

        const raymarchRenderPipelineManager = new GpuRaymarchRenderPipelineManager({
            device,
            format,
            depthFormat: "depth24plus",
            uniformsManager,
            mpmManager,
        });
        this.raymarchRenderPipelineManager = raymarchRenderPipelineManager;

        const mpmGridRenderPipelineManager = new GpuMpmGridRenderPipelineManager({
            device,
            format,
            depthFormat: "depth24plus",
            uniformsManager,
            mpmManager,
        });
        this.mpmGridRenderPipelineManager = mpmGridRenderPipelineManager;

        const volumetricBufferManager = new GpuVolumetricBufferManager({
            device,
            gridResolutionX,
            gridResolutionY,
            gridResolutionZ,
            screenDims: { width: camera.screenDims.width(), height: camera.screenDims.height() },
        });
        this.volumetricBufferManager = volumetricBufferManager;

        const volumetricRenderPipelineManager = new GpuVolumetricRenderPipelineManager({
            device,
            format,
            uniformsManager,
            volumetricBufferManager,
            mpmBufferManager: mpmManager,
        });
        this.volumetricRenderPipelineManager = volumetricRenderPipelineManager;


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

        this.volumetricRenderPipelineManager.resize(this.device, width, height);
    }

    scatterParticlesInMeshVolume() {
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
    }

    updateColliderVel(transform: [number, number, number]) {
        this.uniformsManager.writeColliderVel(transform);
        // this.v = transform;
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
            this.mpmPipelineManager.addDispatch({
                computePassEncoder,
                pipeline: this.mpmPipelineManager.gridClearComputePipeline,
                dispatchX: Math.ceil(this.gridResolutionX / 8),
                dispatchY: Math.ceil(this.gridResolutionY / 8),
                dispatchZ: Math.ceil(this.gridResolutionZ / 4),
            });
            
            this.mpmPipelineManager.addDispatch({
                computePassEncoder,
                dispatchX: Math.ceil(this.nParticles / 256),
                pipeline: this.mpmPipelineManager.p2gComputePipeline,
            });

            this.mpmPipelineManager.addDispatch({
                computePassEncoder,
                pipeline: this.mpmPipelineManager.gridComputePipeline,
                dispatchX: Math.ceil(this.gridResolutionX / 8),
                dispatchY: Math.ceil(this.gridResolutionY / 8),
                dispatchZ: Math.ceil(this.gridResolutionZ / 4),
            });

            this.mpmPipelineManager.addDispatch({
                computePassEncoder,
                dispatchX: Math.ceil(this.nParticles / 256),
                pipeline: this.mpmPipelineManager.g2pComputePipeline,
            });
        }

        computePassEncoder.end();
    }

    async addRenderPass(commandEncoder: GPUCommandEncoder) {
        this.uniformsManager.writeViewProjMat(this.camera.viewProjMat);
        this.uniformsManager.writeViewProjInvMat(this.camera.viewProjInvMat);

        if (this.getRenderMethodType() === GpuRenderMethodType.Volumetric) {
            commandEncoder.clearBuffer(this.volumetricBufferManager.massGridBuffer);

            const volComputePass = commandEncoder.beginComputePass({
                label: "volumetric compute pass",
            });

            this.volumetricRenderPipelineManager.addMassCalulationDispatch(volComputePass, this.nParticles);
            this.volumetricRenderPipelineManager.addRaymarchDispatch(volComputePass);
            
            volComputePass.end();
        }

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
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
                depthClearValue: 1.0,
            },
            timestampWrites: this.performanceMeasurementManager !== null
                ? {
                    querySet: this.performanceMeasurementManager.querySet,
                    beginningOfPassWriteIndex: 0,
                    endOfPassWriteIndex: 1,
                }
                : undefined,
        });

        this.selectRenderPipelineManager().addDraw(renderPassEncoder);
        if (this.getRenderMethodType() !== GpuRenderMethodType.Volumetric) {
            this.rasterizeRenderPipelineManager.addDraw(renderPassEncoder);
            this.mpmGridRenderPipelineManager.addDraw(renderPassEncoder);
        }

        renderPassEncoder.end();
    }

    loop({
        onGpuTimeUpdate,
        onAnimationFrameTimeUpdate,
        onUserControlUpdate,
    }: {
        onGpuTimeUpdate?: (ns: bigint) => void,
        onAnimationFrameTimeUpdate?: (ms: number) => void,
        onUserControlUpdate?: () => void,
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

            onUserControlUpdate?.();

            const commandEncoder = this.device.createCommandEncoder({
                label: "loop command encoder",
            });

            // debug
            // commandEncoder.copyBufferToBuffer(
            //     this.rasterizeRenderPipelineManager.colliderManager.indicesBuffer, 0,
            //     this.readbackBuffer, 0,
            //     this.rasterizeRenderPipelineManager.colliderManager.indicesBuffer.size
            // );
            // await this.readbackBuffer.mapAsync(GPUMapMode.READ);
            // const data = new Uint32Array(this.readbackBuffer.getMappedRange());
            // console.log(data);
            // this.readbackBuffer.unmap();
            // console.log(this.v);

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

            case GpuRenderMethodType.Volumetric:
                return this.volumetricRenderPipelineManager;

        }
    }
}