import type { Camera } from "$lib/Camera.svelte";
import { GpuPointsRenderPipelineManager } from "./pipelines/GpuPointsRenderPipelineManager";
import { GpuSimulationStepPipelineManager } from "./pipelines/GpuSimulationStepPipelineManager";
import { GpuUniformsBufferManager } from "./buffers/GpuUniformsBufferManager";
import { GpuMpmBufferManager } from "./buffers/GpuMpmBufferManager";
import { GpuRaymarchRenderPipelineManager } from "./pipelines/GpuRaymarchRenderPipelineManager";

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
    private readonly mpmManager: GpuMpmBufferManager;
    private readonly simulationStepPipelineManager: GpuSimulationStepPipelineManager;
    private readonly pointsRenderPipelineManager: GpuPointsRenderPipelineManager;
    private readonly raymarchRenderPipelineManager: GpuRaymarchRenderPipelineManager;

    constructor({
        device,
        format,
        context,
        nParticles,
        gridResolution,
        simulationTimestepS,
        camera,
        initialPositions,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        context: GPUCanvasContext,
        nParticles: number,
        gridResolution: number,
        simulationTimestepS: number,
        camera: Camera,
        initialPositions?: Float32Array,
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
        this.mpmManager = mpmManager;

        const simulationStepPipelineManager = new GpuSimulationStepPipelineManager({
            device,
            particleDataBuffer: mpmManager.particleDataBuffer,
            gridDataBuffer: mpmManager.gridDataBuffer,
            uniformsManager,
        });
        this.simulationStepPipelineManager = simulationStepPipelineManager;

        const pointsRenderPipelineManager = new GpuPointsRenderPipelineManager({device, format, uniformsManager});
        this.pointsRenderPipelineManager = pointsRenderPipelineManager;

        const raymarchRenderPipelineManager = new GpuRaymarchRenderPipelineManager({device, format, uniformsManager, mpmManager});
        this.raymarchRenderPipelineManager = raymarchRenderPipelineManager;
    }

    async doSimulationSteps(nSteps: number) {
        const commandEncoder = this.device.createCommandEncoder({
            label: "simulation step command encoder",
        });

        const computePassEncoder = commandEncoder.beginComputePass();
        
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

        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }

    async render() {
        this.uniformsManager.writeViewInvProjMat(this.camera.viewInvProj);
        this.uniformsManager.writeViewInvMat(this.camera.viewInv);
        
        const commandEncoder = this.device.createCommandEncoder({
            label: "render command encoder",
        });
        // this.pointsRenderPipelineManager.addRenderPass({
        //     commandEncoder,
        //     context: this.context,
        //     particleDataBuffer: this.mpmManager.particleDataBuffer,
        //     nParticles: this.nParticles,
        // });

        this.raymarchRenderPipelineManager.addRenderPass({
            commandEncoder,
            context: this.context,
        });

        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }

    loop() {
        let handle = 0;
        let canceled = false;

        const simulationTimestepMs = this.simulationTimestepS * 1_000;


        let nSimulationStep = 0;
        let simulationStartTime = Date.now();
        const loop = async () => {
            // catch up the simulation to the current time
            let currentSimulationTime = simulationStartTime + nSimulationStep * simulationTimestepMs;
            let timeToSimulate = Date.now() - currentSimulationTime;

            const nSteps = Math.ceil(timeToSimulate / simulationTimestepMs);
            // if drifting too much, drop simulation steps 
            if (timeToSimulate <= MAX_SIMULATION_DRIFT_MS) {
                await this.doSimulationSteps(nSteps);
            }
            nSimulationStep += nSteps;


            await this.render();

            if (canceled) return;
            handle = requestAnimationFrame(loop);
        };

        handle = requestAnimationFrame(loop);

        return () => {
            cancelAnimationFrame(handle);
            canceled = true;
        };
    }
}