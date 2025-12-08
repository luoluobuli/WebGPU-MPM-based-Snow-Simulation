
export class GpuDepthPicker {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private readonly outputBuffer: GPUBuffer;
    private readonly coordsBuffer: GPUBuffer;

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

        const shader = `
@group(0) @binding(0) var depthTex: texture_depth_2d;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> coords: vec2<u32>;

@compute @workgroup_size(1)
fn main() {
    output[0] = textureLoad(depthTex, vec2<i32>(coords), 0);
}
`;

        this.pipeline = device.createComputePipeline({
            label: "DepthPicker pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [this.bindGroupLayout],
            }),
            compute: {
                module: device.createShaderModule({ code: shader }),
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
        // Validation?
        // x, y are integers.
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

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const array = new Float32Array(readbackBuffer.getMappedRange());
        const depth = array[0];
        readbackBuffer.unmap();
        readbackBuffer.destroy();

        return depth;
    }

    destroy() {
        this.outputBuffer.destroy();
        this.coordsBuffer.destroy();
    }
}
