import type { Camera } from "$lib/Camera.svelte";
import { GpuPointsRenderPipelineManager } from "./pipelines/GpuPointsRenderPipelineManager";
import { GpuSimulationStepPipelineManager } from "./pipelines/GpuSimulationStepPipelineManager";
import { GpuUniformsBufferManager } from "./buffers/GpuUniformsBufferManager";
import { GpuMpmBufferManager } from "./buffers/GpuMpmBufferManager";
import { GpuRaymarchRenderPipelineManager } from "./pipelines/GpuRaymarchRenderPipelineManager";
import { GpuRenderMethodType, type GpuRenderMethod } from "./pipelines/GpuRenderMethod";
import { GpuPerformanceMeasurementBufferManager } from "./buffers/GpuPerformanceMeasurementBufferManager";

const MAX_SIMULATION_DRIFT_MS = 1_000;
const FP_SCALE = 1024.0;

export class GpuSnowPipelineRunner {
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly nParticles: number;
    private readonly gridResolution: number;
    private readonly simulationTimestepS: number;
    private readonly camera: Camera;

    private readonly uniformsManager: GpuUniformsBufferManager;
    private readonly performanceMeasurementManager: GpuPerformanceMeasurementBufferManager | null;
    private readonly simulationStepPipelineManager: GpuSimulationStepPipelineManager;
    private readonly pointsRenderPipelineManager: GpuPointsRenderPipelineManager;
    private readonly raymarchRenderPipelineManager: GpuRaymarchRenderPipelineManager;

    private readonly getRenderMethodType: () => GpuRenderMethodType;

    constructor({
        device,
        format,
        context,
        nParticles,
        gridResolution,
        simulationTimestepS,
        camera,
        initialPositions,
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
        initialPositions?: Float32Array,
        getRenderMethodType: () => GpuRenderMethodType,
        measurePerf: boolean,
    }) {
        this.device = device;
        this.context = context;
        this.nParticles = nParticles;
        this.gridResolution = gridResolution;
        this.simulationTimestepS = simulationTimestepS;

        this.camera = camera;

        const uniformsManager = new GpuUniformsBufferManager({device});
        this.uniformsManager = uniformsManager;

        uniformsManager.writeSimulationTimestepS(simulationTimestepS);
        uniformsManager.writeGridResolution(gridResolution);
        uniformsManager.writeFixedPointScale(FP_SCALE);
        uniformsManager.writeGridMinCoords([-2, -2, 0]);
        uniformsManager.writeGridMaxCoords([2, 2, 4]);

        const mpmManager = new GpuMpmBufferManager({device, nParticles, gridResolution, initialPositions});

        const simulationStepPipelineManager = new GpuSimulationStepPipelineManager({
            device,
            particleDataBuffer: mpmManager.particleDataBuffer,
            gridDataBuffer: mpmManager.gridDataBuffer,
            uniformsManager,
        });
        this.simulationStepPipelineManager = simulationStepPipelineManager;

        const pointsRenderPipelineManager = new GpuPointsRenderPipelineManager({device, format, uniformsManager, mpmManager});
        this.pointsRenderPipelineManager = pointsRenderPipelineManager;

        const raymarchRenderPipelineManager = new GpuRaymarchRenderPipelineManager({device, format, uniformsManager, mpmManager});
        this.raymarchRenderPipelineManager = raymarchRenderPipelineManager;

        this.getRenderMethodType = getRenderMethodType;

        this.performanceMeasurementManager = measurePerf
            ? new GpuPerformanceMeasurementBufferManager({device})
            : null;

    }

    private async addSimulationStepsComputePass({
        commandEncoder,
        nSteps,
        onGpuElapsedComputeTimeUpdate,
    }: {
        commandEncoder: GPUCommandEncoder,
        nSteps: number,
        onGpuElapsedComputeTimeUpdate?: (gpuElapsedTimeNs: bigint) => void,
    }) {
        const computePassEncoder = commandEncoder.beginComputePass({
            label: "simulation step compute pass",
            // timestampWrites: this.performanceMeasurementManager !== null
            //     ? {
            //         querySet: this.performanceMeasurementManager.querySet,
            //         beginningOfPassWriteIndex: 0,
            //         endOfPassWriteIndex: 1,
            //     }
            //     : undefined,
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

        if (this.performanceMeasurementManager !== null) {
            this.performanceMeasurementManager.addComputeResolve(commandEncoder);
        }
    }

    async addRenderPass(commandEncoder: GPUCommandEncoder) {
        this.uniformsManager.writeViewProjMat(this.camera.viewProjMat);
        this.uniformsManager.writeViewProjInvMat(this.camera.viewProjInvMat);
        
        const renderPassEncoder = commandEncoder.beginRenderPass({
            label: "render pass",
            colorAttachments: [
                {
                    clearValue: {
                        r: 0,
                        g: 0,
                        b: 0,
                        a: 1,
                    },

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

        if (this.performanceMeasurementManager !== null) {
            this.performanceMeasurementManager.addRenderResolve(commandEncoder);
        }
    }

    loop({
        onGpuElapsedComputeTimeUpdate,
        onGpuElapsedRenderTimeUpdate,
    }: {
        onGpuElapsedComputeTimeUpdate?: (gpuElapsedTimeNs: bigint) => void,
        onGpuElapsedRenderTimeUpdate?: (gpuElapsedTimeNs: bigint) => void,
    } = {}) {
        let handle = 0;
        let canceled = false;

        const simulationTimestepMs = this.simulationTimestepS * 1_000;


        let nSimulationStep = 0;
        let simulationStartTime = Date.now();
        const loop = async () => {
            const commandEncoder = this.device.createCommandEncoder({
                label: "loop command encoder",
            });

            // catch up the simulation to the current time
            let currentSimulationTime = simulationStartTime + nSimulationStep * simulationTimestepMs;
            let timeToSimulate = Date.now() - currentSimulationTime;

            const nSteps = Math.ceil(timeToSimulate / simulationTimestepMs);
            // if drifting too much, drop simulation steps 
            if (timeToSimulate <= MAX_SIMULATION_DRIFT_MS) {
                this.addSimulationStepsComputePass({
                    commandEncoder,
                    nSteps,
                    onGpuElapsedComputeTimeUpdate,
                });
            }
            nSimulationStep += nSteps;


            this.addRenderPass(commandEncoder);

            this.device.queue.submit([commandEncoder.finish()]);
            // await this.device.queue.onSubmittedWorkDone();

            if (this.performanceMeasurementManager !== null) {
                this.performanceMeasurementManager.mapGpuElapsedComputeTimeNs()
                    .then(elapsedTimeNs => {
                        if (elapsedTimeNs === null) return;
                        onGpuElapsedComputeTimeUpdate?.(elapsedTimeNs);
                    });

                this.performanceMeasurementManager.mapGpuElapsedRenderTimeNs()
                    .then(elapsedTimeNs => {
                        if (elapsedTimeNs === null) return;
                        onGpuElapsedRenderTimeUpdate?.(elapsedTimeNs);
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