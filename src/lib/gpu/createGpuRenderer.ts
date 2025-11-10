export const createGpuRenderer = ({
    device,
    context,
    nParticles,
    renderBindGroup,
    renderPipeline,
    particlePosBuffer,
}: {
    device: GPUDevice,
    context: WebGPURenderingContext,
    nParticles: number,
    renderBindGroup: GPUBindGroup,
    renderPipeline: GPURenderPipeline,
    particlePosBuffer: GPUBuffer,
}) => {
    return async () => {
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