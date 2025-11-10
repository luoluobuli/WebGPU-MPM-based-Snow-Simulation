import type { Camera } from "$lib/Camera.svelte";
import { setupGpuPipelines } from "./setupGpuPipelines";

const MAX_SIMULATION_DRIFT_MS = 1_000;

export class GpuSnowPipelineRunner {
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly nParticles: number;
    private readonly simulationTimestepS: number;
    private readonly camera: Camera;

    private buffer1IsSource = true;

    private readonly pipelineData: ReturnType<typeof setupGpuPipelines>;

    constructor({
        device,
        format,
        context,
        nParticles,
        simulationTimestepS,
        camera,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        context: GPUCanvasContext,
        nParticles: number,
        simulationTimestepS: number,
        camera: Camera,
    }) {
        this.device = device;
        this.context = context;
        this.nParticles = nParticles;
        this.simulationTimestepS = simulationTimestepS;

        this.camera = camera;

        this.pipelineData = setupGpuPipelines({device, format, nParticles});
    }

    async doSimulationStep() {
        const commandEncoder = this.device.createCommandEncoder({
            label: "simulation step command encoder",
        });

        const computePassEncoder = commandEncoder.beginComputePass({
            label: "simulation step compute pass",
        });
        computePassEncoder.setPipeline(this.pipelineData.simulationStepPipeline);
        computePassEncoder.setBindGroup(0, this.pipelineData.uniformsBindGroup);
        computePassEncoder.setBindGroup(1, this.simulationStepStorageBindGroup);
        computePassEncoder.dispatchWorkgroups(Math.ceil(this.nParticles / 256));
        computePassEncoder.end();


        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        this.buffer1IsSource = !this.buffer1IsSource;
    }

    async render() {
        this.device.queue.writeBuffer(this.pipelineData.uniformsBuffer, 0, new Float32Array([this.simulationTimestepS]));
        this.device.queue.writeBuffer(this.pipelineData.uniformsBuffer, 16, this.camera.viewInvProj.buffer);
        
        const commandEncoder = this.device.createCommandEncoder({
            label: "render command encoder",
        });

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
        });
        renderPassEncoder.setBindGroup(0, this.pipelineData.uniformsBindGroup);
        renderPassEncoder.setVertexBuffer(0, this.particleDataBuffer);
        renderPassEncoder.setPipeline(this.pipelineData.renderPipeline);
        renderPassEncoder.draw(this.nParticles);
        renderPassEncoder.end();


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

    private get simulationStepStorageBindGroup() {
        return this.buffer1IsSource
            ? this.pipelineData.simulationStepStorageBindGroup1_2
            : this.pipelineData.simulationStepStorageBindGroup2_1;
    }

    private get particleDataBuffer() {
        return this.buffer1IsSource
            ? this.pipelineData.particleDataBuffer1
            : this.pipelineData.particleDataBuffer2;
    }
}