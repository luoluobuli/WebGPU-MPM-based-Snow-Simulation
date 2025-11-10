import type { Camera } from "$lib/Camera.svelte";
import { setupGpuPipelines } from "./setupGpuPipelines";

export class GpuSnowPipelineRunner {
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly nParticles: number;
    private readonly camera: Camera;

    private buffer1IsSource = true;

    private readonly pipelineData: ReturnType<typeof setupGpuPipelines>;

    constructor({
        device,
        format,
        context,
        nParticles,
        camera,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        context: GPUCanvasContext,
        nParticles: number,
        camera: Camera,
    }) {
        this.device = device;
        this.context = context;
        this.nParticles = nParticles;

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
        this.device.queue.writeBuffer(this.pipelineData.uniformsBuffer, 0, this.camera.viewInvProj.buffer);
        
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