import type { GpuUniformsBufferManager } from "$lib/gpu/buffers/GpuUniformsBufferManager";
import commonModuleSrc from "$lib/gpu/shaders/_common.wgsl?raw";
import pointsVertexModuleSrc from "$lib/gpu/shaders/pointsVertex.wgsl?raw";
import pointsFragmentModuleSrc from "$lib/gpu/shaders/pointsFragment.wgsl?raw";
import type { GpuMpmBufferManager } from "../buffers/GpuMpmBufferManager";
import type { GpuRenderMethod } from "./GpuRenderMethod";

export class GpuPointsRenderPipelineManager implements GpuRenderMethod {
    readonly renderPipeline: GPURenderPipeline;

    readonly uniformsManager: GpuUniformsBufferManager;
    readonly mpmManager: GpuMpmBufferManager;

    constructor({
        device,
        format,
        uniformsManager,
        mpmManager,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        uniformsManager: GpuUniformsBufferManager,
        mpmManager: GpuMpmBufferManager,
    }) {
        const vertexModule = device.createShaderModule({
            label: "points vertex module",
            code: commonModuleSrc + pointsVertexModuleSrc,
        });
        const fragmentModule = device.createShaderModule({
            label: "points fragment module",
            code: commonModuleSrc + pointsFragmentModuleSrc,
        });
        
        const renderPipelineLayout = device.createPipelineLayout({
            label: "points render pipeline",
            bindGroupLayouts: [uniformsManager.bindGroupLayout],
        });
        this.renderPipeline = device.createRenderPipeline({
            label: "points render pipeline",

            layout: renderPipelineLayout,

            vertex: {
                module: vertexModule,
                entryPoint: "vert",
                buffers: [
                    {
                        attributes: [
                            {
                                shaderLocation: 0,
                                offset: 0,
                                format: "float32x4",
                            },
                        ],
                        arrayStride: 96,
                        stepMode: "vertex",
                    },
                ],
            },

            fragment: {
                module: fragmentModule,
                entryPoint: "frag",
                targets: [
                    {
                        format,
                    },
                ],
            },

            primitive: {
                topology: "point-list",
            },
        });

        this.uniformsManager = uniformsManager;
        this.mpmManager = mpmManager;
    }

    addDraw(renderPassEncoder: GPURenderPassEncoder) {
        renderPassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        renderPassEncoder.setVertexBuffer(0, this.mpmManager.particleDataBuffer);
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.draw(this.mpmManager.nParticles);
    }
}