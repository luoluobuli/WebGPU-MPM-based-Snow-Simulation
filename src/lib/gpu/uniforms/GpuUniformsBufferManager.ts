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
            size: 336,
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

    writeFixedPointScale(fixedPointScale: number) {
        this.device.queue.writeBuffer(this.buffer, 4, new Float32Array([fixedPointScale]));
    }

    writeUsePbmpm(usePbmpm: number) {
        this.device.queue.writeBuffer(this.buffer, 8, new Uint32Array([usePbmpm]));
    }

    writeGridMinCoords(min: [number, number, number]) {
        this.device.queue.writeBuffer(this.buffer, 16, new Float32Array(min));
    }

    writeGridMaxCoords(max: [number, number, number]) {
        this.device.queue.writeBuffer(this.buffer, 32, new Float32Array(max));
    }

    writeViewProjMat(viewProjMat: Mat4) {
        this.device.queue.writeBuffer(this.buffer, 48, viewProjMat.buffer);
    }

    writeViewProjInvMat(viewProjInvMat: Mat4) {
        this.device.queue.writeBuffer(this.buffer, 112, viewProjInvMat.buffer);
    }

    writeMeshMinCoords(min: [number, number, number]) {
        this.device.queue.writeBuffer(this.buffer, 176, new Float32Array(min));
    }

    writeMeshMaxCoords(max: [number, number, number]) {
        this.device.queue.writeBuffer(this.buffer, 192, new Float32Array(max));
    }

    writeGridResolution(gridResolution: [number, number, number]) {
        this.device.queue.writeBuffer(this.buffer, 208, new Uint32Array(gridResolution));
    }

    writeColliderMinCoords(min: [number, number, number]) {
        this.device.queue.writeBuffer(this.buffer, 224, new Float32Array(min));
    }

    writeColliderMaxCoords(max: [number, number, number]) {
        this.device.queue.writeBuffer(this.buffer, 240, new Float32Array(max));
    }

    writeColliderTransformMat(transformMat: Mat4) {
        this.device.queue.writeBuffer(this.buffer, 256, transformMat.buffer);
    }

    writeColliderVel(vel: [number, number, number]) {
        this.device.queue.writeBuffer(this.buffer, 320, new Float32Array(vel));
    }
}