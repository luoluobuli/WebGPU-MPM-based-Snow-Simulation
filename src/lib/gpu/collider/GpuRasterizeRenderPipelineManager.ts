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
    readonly textureBindGroup: GPUBindGroup;
    readonly textureBindGroupLayout: GPUBindGroupLayout;


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
        
        // Create texture bind group layout
        this.textureBindGroupLayout = device.createBindGroupLayout({
            label: "collider texture bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: { type: "filtering" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "float", viewDimension: "2d-array" },
                },
            ],
        });
        
        // Create texture bind group
        this.textureBindGroup = device.createBindGroup({
            label: "collider texture bind group",
            layout: this.textureBindGroupLayout,
            entries: [
                { binding: 0, resource: colliderManager.sampler },
                { binding: 1, resource: colliderManager.textureArrayView! },
            ],
        });
        
        const renderPipelineLayout = device.createPipelineLayout({
            label: "rasterize render pipeline layout",
            bindGroupLayouts: [
                uniformsManager.bindGroupLayout,
                this.textureBindGroupLayout,
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
                    { // UVs
                        attributes: [
                            {
                                shaderLocation: 2,
                                offset: 0,
                                format: "float32x2",
                            },
                        ],
                        arrayStride: 8,
                        stepMode: "vertex",
                    },
                    { // Material indices
                        attributes: [
                            {
                                shaderLocation: 3,
                                offset: 0,
                                format: "uint32",
                            },
                        ],
                        arrayStride: 4,
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
        renderPassEncoder.setBindGroup(1, this.textureBindGroup);
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.setVertexBuffer(0, this.colliderManager.colliderDataBuffer, this.colliderManager.verticesOffset);
        renderPassEncoder.setVertexBuffer(1, this.colliderManager.colliderDataBuffer, this.colliderManager.normalsOffset);
        renderPassEncoder.setVertexBuffer(2, this.colliderManager.colliderDataBuffer, this.colliderManager.uvsOffset);
        renderPassEncoder.setVertexBuffer(3, this.colliderManager.colliderDataBuffer, this.colliderManager.materialIndicesOffset);
        renderPassEncoder.setIndexBuffer(this.colliderManager.colliderDataBuffer, "uint32", this.colliderManager.indicesOffset);
        renderPassEncoder.drawIndexed(this.indexCount);
    }
}
