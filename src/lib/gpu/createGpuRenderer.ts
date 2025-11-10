import type { Camera } from "$lib/Camera.svelte";

export const createGpuRenderer = ({
    device,
    context,
    renderPipeline,
    renderBindGroup,
    
    particlePosBuffer,
    uniformsBuffer,

    nParticles,
    camera,
}: {
    device: GPUDevice,
    context: GPUCanvasContext,
    renderBindGroup: GPUBindGroup,
    renderPipeline: GPURenderPipeline,

    particlePosBuffer: GPUBuffer,
    uniformsBuffer: GPUBuffer,

    nParticles: number,
    camera: Camera,
}) => {
    return async () => {
        device.queue.writeBuffer(uniformsBuffer, 0, camera.viewInvProj.buffer);
        
        const commandEncoder = device.createCommandEncoder();

        const renderPassEncoder = commandEncoder.beginRenderPass({
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
                    view: context.getCurrentTexture().createView(),
                },
            ],
        });
        renderPassEncoder.setBindGroup(0, renderBindGroup);
        renderPassEncoder.setVertexBuffer(0, particlePosBuffer);
        renderPassEncoder.setPipeline(renderPipeline);
        renderPassEncoder.draw(nParticles);
        renderPassEncoder.end();


        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();
    };
}