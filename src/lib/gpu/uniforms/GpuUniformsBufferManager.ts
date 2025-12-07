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
            size: 37376,
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

    writeUsePbmpm(usePbmpm: boolean) {
        this.device.queue.writeBuffer(this.buffer, 8, new Uint32Array([usePbmpm ? 1 : 0]));
    }

    writeTime(time: number) {
        this.device.queue.writeBuffer(this.buffer, 12, new Uint32Array([time]));
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

    writeCameraPos(cameraPos: [number, number, number]) {
        this.device.queue.writeBuffer(this.buffer, 336, new Float32Array(cameraPos));
    }

    writeGridCellDims(gridCellDims: [number, number, number]) {
        this.device.queue.writeBuffer(this.buffer, 352, new Float32Array(gridCellDims));
    }
    writeColliderNumIndices(numIndices: number) {
        this.device.queue.writeBuffer(this.buffer, 348, new Uint32Array([numIndices]));
    }

    writeColliderNumObjects(numObjects: number) {
        this.device.queue.writeBuffer(this.buffer, 364, new Uint32Array([numObjects]));
    }

    writeLightViewProjMat(lightViewProjMat: Mat4) {
        this.device.queue.writeBuffer(this.buffer, 368, lightViewProjMat.buffer);
    }

    writeColliderTransformInv(transformInv: Mat4) {
        this.device.queue.writeBuffer(this.buffer, 432, transformInv.buffer);
    }

    // Interaction Uniforms (Start 496)
    writeInteractionPos(pos: [number, number, number]) {
        this.device.queue.writeBuffer(this.buffer, 496, new Float32Array(pos));
    }

    writeInteractionStrength(strength: number) {
        this.device.queue.writeBuffer(this.buffer, 508, new Float32Array([strength]));
    }

    writeInteractionRadius(radius: number) {
        this.device.queue.writeBuffer(this.buffer, 512, new Float32Array([radius]));
    }

    writeIsInteracting(isInteracting: boolean) {
        this.device.queue.writeBuffer(this.buffer, 516, new Uint32Array([isInteracting ? 1 : 0]));
    }

    writeColliderObjects(objects: { min: [number, number, number], max: [number, number, number], startIndex: number, countIndices: number }[]) {
        const MAX_OBJECTS = 1024;
        const count = Math.min(objects.length, MAX_OBJECTS);
        const buffer = new ArrayBuffer(count * 32);
        const f32 = new Float32Array(buffer);
        const u32 = new Uint32Array(buffer);

        for (let i = 0; i < count; i++) {
            const obj = objects[i];
            const base = i * 8; // 32 bytes = 8 floats/uints
            f32[base + 0] = obj.min[0];
            f32[base + 1] = obj.min[1];
            f32[base + 2] = obj.min[2];
            u32[base + 3] = obj.startIndex;
            f32[base + 4] = obj.max[0];
            f32[base + 5] = obj.max[1];
            f32[base + 6] = obj.max[2];
            u32[base + 7] = obj.countIndices;
        }
        this.device.queue.writeBuffer(this.buffer, 528, buffer);
    }
}