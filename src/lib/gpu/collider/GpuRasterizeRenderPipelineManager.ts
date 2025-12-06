import type { GpuUniformsBufferManager } from "$lib/gpu/uniforms/GpuUniformsBufferManager";
import type { GpuRenderMethod } from "$lib/gpu/GpuRenderMethod";
import type { GpuColliderBufferManager } from "./GpuColliderBufferManager";

import rasterizeVertexModuleSrc from "./rasterizeVertex.wgsl?raw";
import rasterizeFragmentModuleSrc from "./rasterizeFragment.wgsl?raw";
import { attachPrelude } from "../shaderPrelude";

export class GpuRasterizeRenderPipelineManager {
    readonly renderPipeline: GPURenderPipeline;
    readonly indexCount : number;

    readonly uniformsManager: GpuUniformsBufferManager;
    readonly colliderManager: GpuColliderBufferManager;


    constructor({
        device,
        format,
        depthFormat = "depth24plus",
        uniformsManager,
        colliderManager,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        depthFormat?: GPUTextureFormat,
        uniformsManager: GpuUniformsBufferManager,
        colliderManager: GpuColliderBufferManager, 
    }) {
        const vertexModule = device.createShaderModule({
            label: "rasterize vertex module",
            code: attachPrelude(rasterizeVertexModuleSrc),
        });

        const fragmentModule = device.createShaderModule({
            label: "rasterize fragment module",
            code: attachPrelude(rasterizeFragmentModuleSrc),
        });
        
        const renderPipelineLayout = device.createPipelineLayout({
            label: "rasterize render pipeline layout",
            bindGroupLayouts: [
                uniformsManager.bindGroupLayout,
            ],
        });

        this.renderPipeline = device.createRenderPipeline({
            label: "rasterize render pipeline",
            layout: renderPipelineLayout,

            vertex: {
                module: vertexModule,
                entryPoint: "vert",
                buffers: [
                    { // positions
                        attributes: [
                            { 
                                shaderLocation: 0,
                                offset: 0,
                                format: "float32x3",
                            },
                        ],
                        arrayStride: 12,
                        stepMode: "vertex",
                    },
                    { // normals
                        attributes: [
                            {
                                shaderLocation: 1,
                                offset: 0,
                                format: "float32x3",
                            },
                        ],
                        arrayStride: 12,
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
                topology: "triangle-list",
                frontFace: "cw",
                cullMode: "back",
            },

            depthStencil: {
                format: depthFormat,
                depthWriteEnabled: true,
                depthCompare: "less",
            },
        });

        this.uniformsManager = uniformsManager;
        this.colliderManager = colliderManager;

        this.indexCount = colliderManager.numIndices;
    }

    addDraw(renderPassEncoder: GPURenderPassEncoder) {
        renderPassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.setVertexBuffer(0, this.colliderManager.verticesBuffer);
        renderPassEncoder.setVertexBuffer(1, this.colliderManager.normalsBuffer);
        renderPassEncoder.setIndexBuffer(this.colliderManager.indicesBuffer, "uint32");
        renderPassEncoder.drawIndexed(this.indexCount);
    }
}