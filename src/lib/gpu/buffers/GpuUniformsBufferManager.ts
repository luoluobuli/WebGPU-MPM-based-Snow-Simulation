import type { Mat4 } from "wgpu-matrix";

export class GpuUniformsBufferManager {
    private readonly device: GPUDevice;

    readonly buffer: GPUBuffer;
    readonly bindGroupLayout: GPUBindGroupLayout;
    readonly bindGroup: GPUBindGroup;

    constructor({
        device,
    }: {
        device: GPUDevice,
    }) {
        const uniformsBuffer = device.createBuffer({
            label: "uniforms buffer",
            size: 176,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
        });

        const uniformsBindGroupLayout = device.createBindGroupLayout({
            label: "uniforms bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform",
                    },
                },
            ],
        });
        const uniformsBindGroup = device.createBindGroup({
            label: "uniforms bind group",
            layout: uniformsBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: uniformsBuffer,
                    },
                },
            ],
        });


        this.device = device;

        this.buffer = uniformsBuffer;
        this.bindGroupLayout = uniformsBindGroupLayout;
        this.bindGroup = uniformsBindGroup;
    }

    writeSimulationTimestepS(timestep: number) {
        this.device.queue.writeBuffer(this.buffer, 0, new Float32Array([timestep]));
    }

    writeGridResolution(gridResolution: number) {
        this.device.queue.writeBuffer(this.buffer, 4, new Uint32Array([gridResolution]));
    }

    writeFixedPointScale(fixedPointScale: number) {
        this.device.queue.writeBuffer(this.buffer, 8, new Float32Array([fixedPointScale]));
    }

    writeGridMinCoords(min: [number, number, number]) {
        this.device.queue.writeBuffer(this.buffer, 16, new Float32Array(min));
    }

    writeGridMaxCoords(max: [number, number, number]) {
        this.device.queue.writeBuffer(this.buffer, 32, new Float32Array(max));
    }

    writeViewInvProjMat(viewInvProjMat: Mat4) {
        this.device.queue.writeBuffer(this.buffer, 48, viewInvProjMat.buffer);
    }

    writeViewInvMat(viewInvMat: Mat4) {
        this.device.queue.writeBuffer(this.buffer, 112, viewInvMat.buffer);
    }
}