import type { GpuUniformsBufferManager } from "$lib/gpu/uniforms/GpuUniformsBufferManager";
import type { GpuRenderMethod } from "$lib/gpu/GpuRenderMethod";
import type { GpuColliderBufferManager } from "./GpuColliderBufferManager";

import rasterizeVertexModuleSrc from "./rasterizeVertex.wgsl?raw";
import rasterizeFragmentModuleSrc from "./rasterizeFragment.wgsl?raw";
import { attachPrelude } from "../shaderPrelude";

export class GpuRasterizeRenderPipelineManager implements GpuRenderMethod {
    readonly renderPipeline: GPURenderPipeline;
    readonly rasterizeStorageBindGroup: GPUBindGroup;
    readonly indexCount : number;

    readonly uniformsManager: GpuUniformsBufferManager;


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
        const rasterizeStorageBindGroupLayout = device.createBindGroupLayout({
            label: "rasterize storage bind group layout",
            
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage" },
                },
            ],
        });
        
        const rasterizeStorageBindGroup = device.createBindGroup({
            label: "rasterize storage bind group",

            layout: rasterizeStorageBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: colliderManager.colliderVerticesBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: colliderManager.colliderIndicesBuffer,
                    },
                },
            ],
        });


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
                rasterizeStorageBindGroupLayout,
            ],
        });

        this.renderPipeline = device.createRenderPipeline({
            label: "rasterize render pipeline",
            layout: renderPipelineLayout,

            vertex: {
                module: vertexModule,
                entryPoint: "vert",
                buffers: [
                    // {
                    //     attributes: [
                    //         {
                    //             shaderLocation: 0,
                    //             offset: 0,
                    //             format: "float32x3",
                    //         },
                    //     ],
                    //     arrayStride: 12,
                    //     stepMode: "vertex",
                    // },
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
                topology: "line-list",
                cullMode: "back",
            },

            depthStencil: {
                format: depthFormat,
                depthWriteEnabled: true,
                depthCompare: "less",
            },
        });

        this.uniformsManager = uniformsManager;
        this.rasterizeStorageBindGroup = rasterizeStorageBindGroup;

        this.indexCount = colliderManager.numIndices;
    }

    addDraw(renderPassEncoder: GPURenderPassEncoder) {
        renderPassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        renderPassEncoder.setBindGroup(1, this.rasterizeStorageBindGroup);
        renderPassEncoder.setPipeline(this.renderPipeline);
        // renderPassEncoder.draw(this.indexCount);
        renderPassEncoder.draw(24);
    }
}