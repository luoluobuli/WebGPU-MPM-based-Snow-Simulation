import loadDepthModuleSrc from "./load_depth.wgsl?raw";

export class GpuDepthPicker {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private readonly outputBuffer: GPUBuffer;
    private readonly coordsBuffer: GPUBuffer;
    private destroyed = false;

    constructor({ device }: { device: GPUDevice }) {
        this.device = device;

        this.bindGroupLayout = device.createBindGroupLayout({
            label: "DepthPicker bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "depth" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" },
                },
            ],
        });


        this.pipeline = device.createComputePipeline({
            label: "DepthPicker pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [this.bindGroupLayout],
            }),
            compute: {
                module: device.createShaderModule({ code: loadDepthModuleSrc }),
                entryPoint: "main",
            },
        });

        this.outputBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        this.coordsBuffer = device.createBuffer({
            size: 8, // 2 * u32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    async pick(depthTextureView: GPUTextureView, x: number, y: number): Promise<number | null> {
        if (this.destroyed) return null;

        const ix = Math.floor(x);
        const iy = Math.floor(y);

        this.device.queue.writeBuffer(this.coordsBuffer, 0, new Uint32Array([ix, iy]));

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: depthTextureView },
                { binding: 1, resource: { buffer: this.outputBuffer } },
                { binding: 2, resource: { buffer: this.coordsBuffer } },
            ],
        });

        const commandEncoder = this.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();

        // Copy to readback buffer
        const readbackBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        commandEncoder.copyBufferToBuffer(this.outputBuffer, 0, readbackBuffer, 0, 4);
        
        this.device.queue.submit([commandEncoder.finish()]);

        let mapped = false;
        try {
            await readbackBuffer.mapAsync(GPUMapMode.READ);
            mapped = true;

            if (this.destroyed) return null;

            const array = new Float32Array(readbackBuffer.getMappedRange());
            return array[0];
        } catch (error) {
            if (this.destroyed) return null;

            throw error;
        } finally {
            if (mapped && readbackBuffer.mapState === "mapped") {
                readbackBuffer.unmap();
            }
            readbackBuffer.destroy();
        }
    }

    destroy() {
        if (this.destroyed) return;

        this.destroyed = true;
        this.outputBuffer.destroy();
        this.coordsBuffer.destroy();
    }
}
