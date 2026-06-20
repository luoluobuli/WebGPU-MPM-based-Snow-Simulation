import type { GpuColliderBufferManager } from "./GpuColliderBufferManager";
import colliderSdfModuleSrc from "./colliderSdf.cs.wgsl?raw";

const WORKGROUP_SIZE = 64;
const PARAMS_BYTE_LENGTH = 32;

export class GpuColliderSdfPipelineManager {
    private readonly device: GPUDevice;
    private readonly paramsBuffer: GPUBuffer;
    private readonly paramsArrayBuffer = new ArrayBuffer(PARAMS_BYTE_LENGTH);
    private readonly paramsFloat32 = new Float32Array(this.paramsArrayBuffer);
    private readonly paramsUint32 = new Uint32Array(this.paramsArrayBuffer);
    private readonly bindGroup: GPUBindGroup;
    private readonly computePipeline: GPUComputePipeline;

    constructor({
        device,
        colliderManager,
    }: {
        device: GPUDevice,
        colliderManager: GpuColliderBufferManager,
    }) {
        const storageBindGroupLayout = device.createBindGroupLayout({
            label: "collider SDF creation bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage",
                    },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage",
                    },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform",
                    },
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage",
                    },
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage",
                    },
                },
            ],
        });

        const paramsBuffer = device.createBuffer({
            label: "collider SDF creation params buffer",
            size: PARAMS_BYTE_LENGTH,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const bindGroup = device.createBindGroup({
            label: "collider SDF creation bind group",
            layout: storageBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: colliderManager.colliderIndexStorageBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: colliderManager.colliderVertexStorageBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: colliderManager.colliderSdfBuffer,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: paramsBuffer,
                    },
                },
                {
                    binding: 4,
                    resource: {
                        buffer: colliderManager.colliderSdfBvhNodeBuffer,
                    },
                },
                {
                    binding: 5,
                    resource: {
                        buffer: colliderManager.colliderSdfBvhTriangleOrderBuffer,
                    },
                },
            ],
        });
        const pipelineLayout = device.createPipelineLayout({
            label: "collider SDF creation pipeline layout",
            bindGroupLayouts: [
                storageBindGroupLayout,
            ],
        });
        const shaderModule = device.createShaderModule({
            label: "collider SDF creation shader",
            code: colliderSdfModuleSrc,
        });
        const computePipeline = device.createComputePipeline({
            label: "collider SDF creation compute pipeline",
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: "createColliderSdf",
            },
        });

        this.device = device;
        this.paramsBuffer = paramsBuffer;
        this.bindGroup = bindGroup;
        this.computePipeline = computePipeline;
    }

    addDispatch({
        commandEncoder,
        colliderManager,
        timestampWrites,
    }: {
        commandEncoder: GPUCommandEncoder,
        colliderManager: GpuColliderBufferManager,
        timestampWrites?: GPUComputePassTimestampWrites,
    }) {
        if (colliderManager.numIndices === 0 || colliderManager.numSdfBvhNodes === 0) return;

        this.writeParams(colliderManager);

        const computePassEncoder = commandEncoder.beginComputePass({
            label: "collider SDF creation compute pass",
            timestampWrites,
        });
        computePassEncoder.setPipeline(this.computePipeline);
        computePassEncoder.setBindGroup(0, this.bindGroup);
        computePassEncoder.dispatchWorkgroups(
            Math.ceil(
                colliderManager.sdfResolution
                * colliderManager.sdfResolution
                * colliderManager.sdfResolution
                / WORKGROUP_SIZE,
            ),
        );
        computePassEncoder.end();
    }

    destroy() {
        this.paramsBuffer.destroy();
    }

    private writeParams(colliderManager: GpuColliderBufferManager) {
        const resolution = colliderManager.sdfResolution;
        const extent: [number, number, number] = [
            colliderManager.maxCoords[0] - colliderManager.minCoords[0],
            colliderManager.maxCoords[1] - colliderManager.minCoords[1],
            colliderManager.maxCoords[2] - colliderManager.minCoords[2],
        ];
        const lastCoord = Math.max(1, resolution - 1);

        this.paramsFloat32[0] = colliderManager.minCoords[0];
        this.paramsFloat32[1] = colliderManager.minCoords[1];
        this.paramsFloat32[2] = colliderManager.minCoords[2];
        this.paramsUint32[3] = resolution;
        this.paramsFloat32[4] = extent[0] / lastCoord;
        this.paramsFloat32[5] = extent[1] / lastCoord;
        this.paramsFloat32[6] = extent[2] / lastCoord;
        this.paramsUint32[7] = colliderManager.numIndices / 3;

        this.device.queue.writeBuffer(
            this.paramsBuffer,
            0,
            this.paramsArrayBuffer,
        );
    }
}