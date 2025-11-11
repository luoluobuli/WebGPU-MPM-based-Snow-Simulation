import type { Camera } from "$lib/Camera.svelte";
import { GpuPointRenderPipelineManager } from "./pipelines/GpuRenderPipelineManager";
import { GpuSimulationStepPipelineManager } from "./pipelines/GpuSimulationStepPipelineManager";
import { GpuUniformsBufferManager } from "./buffers/GpuUniformsBufferManager";
import { GpuMpmBufferManager } from "./buffers/GpuMpmBufferManager";

const MAX_SIMULATION_DRIFT_MS = 1_000;

export class GpuSnowPipelineRunner {
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly nParticles: number;
    private readonly simulationTimestepS: number;
    private readonly camera: Camera;

    private buffer1IsSource = true;

    private readonly uniformsManager: GpuUniformsBufferManager;
    private readonly mpmManager: GpuMpmBufferManager;
    private readonly simulationStepPipelineManager: GpuSimulationStepPipelineManager;
    private readonly pointsRenderPipelineManager: GpuPointRenderPipelineManager;

    constructor({
        device,
        format,
        context,
        nParticles,
        gridResolution,
        simulationTimestepS,
        camera,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        context: GPUCanvasContext,
        nParticles: number,
        gridResolution: number,
        simulationTimestepS: number,
        camera: Camera,
    }) {
        this.device = device;
        this.context = context;
        this.nParticles = nParticles;
        this.simulationTimestepS = simulationTimestepS;

        this.camera = camera;

        const uniformsManager = new GpuUniformsBufferManager({device});
        this.uniformsManager = uniformsManager;

        const mpmManager = new GpuMpmBufferManager({device, nParticles, gridResolution});
        this.mpmManager = mpmManager;

        const simulationStepPipelineManager = new GpuSimulationStepPipelineManager({
            device,
            particleDataBuffer1: mpmManager.particleDataBuffer1,
            particleDataBuffer2: mpmManager.particleDataBuffer2,
            uniformsManager,
        });
        this.simulationStepPipelineManager = simulationStepPipelineManager;

        const pointsRenderPipelineManager = new GpuPointRenderPipelineManager({device, format, uniformsManager});
        this.pointsRenderPipelineManager = pointsRenderPipelineManager;

    }

    async doSimulationStep() {
        const commandEncoder = this.device.createCommandEncoder({
            label: "simulation step command encoder",
        });
        this.simulationStepPipelineManager.addComputePass({
            commandEncoder,
            nParticles: this.nParticles,
            buffer1IsSource: this.buffer1IsSource,
        });
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        this.buffer1IsSource = !this.buffer1IsSource;
    }

    async render() {
        this.uniformsManager.writeSimulationTimestepS(this.simulationTimestepS);
        this.uniformsManager.writeViewProjInvMat(this.camera.viewInvProj);
        
        const commandEncoder = this.device.createCommandEncoder({
            label: "render command encoder",
        });
        this.pointsRenderPipelineManager.addRenderPass({
            commandEncoder,
            context: this.context,
            particleDataBuffer: this.mpmManager.particleDataBufferCurrent(this.buffer1IsSource),
            nParticles: this.nParticles,
        });
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }

    loop() {
        let handle = 0;

        const simulationTimestepMs = this.simulationTimestepS * 1_000;


        let nSimulationStep = 0;
        let simulationStartTime = Date.now();
        const loop = async () => {
            // catch up the simulation to the current time
            let currentSimulationTime = simulationStartTime + nSimulationStep * simulationTimestepMs;
            let timeToSimulate = Date.now() - currentSimulationTime;
            if (timeToSimulate > MAX_SIMULATION_DRIFT_MS) {
                // if drifting too much, drop simulation steps 
                nSimulationStep += Math.ceil(timeToSimulate / simulationTimestepMs);

                currentSimulationTime = simulationStartTime + nSimulationStep * simulationTimestepMs;
                timeToSimulate = Date.now() - currentSimulationTime;
            }
            while (timeToSimulate > 0) {
                await this.doSimulationStep();

                nSimulationStep++;
                timeToSimulate -= simulationTimestepMs;
            }


            await this.render();

            handle = requestAnimationFrame(loop);
        };

        handle = requestAnimationFrame(loop);

        return () => {
            cancelAnimationFrame(handle);
        };
    }
}