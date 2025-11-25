import type { GpuUniformsBufferManager } from "$lib/gpu/buffers/GpuUniformsBufferManager";
import type { GpuRenderMethod } from "./GpuRenderMethod";
import type { GpuStaticMeshBufferManager } from "../buffers/GpuStaticMeshBufferManager";

import rasterizeVertexModuleSrc from "$lib/gpu/shaders/rasterizeVertex.wgsl?raw";
import rasterizeFragmentModuleSrc from "$lib/gpu/shaders/rasterizeFragment.wgsl?raw";
import { attachPrelude } from "../shaders/prelude";

export class GpuRasterizeRenderPipelineManager implements GpuRenderMethod {
    readonly renderPipeline: GPURenderPipeline;
    readonly rasterizeStorageBindGroup: GPUBindGroup;
    readonly indexCount : number;

    readonly uniformsManager: GpuUniformsBufferManager;
    readonly staticMeshManager: GpuStaticMeshBufferManager;


    constructor({
        device,
        format,
        depthFormat = "depth24plus",
        uniformsManager,
        staticMeshManager,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        depthFormat?: GPUTextureFormat,
        uniformsManager: GpuUniformsBufferManager,
        staticMeshManager: GpuStaticMeshBufferManager, 
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
                        buffer: staticMeshManager.verticesBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: staticMeshManager.indicesBuffer,
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
                    {
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
                cullMode: "back",
            },

            depthStencil: {
                format: depthFormat,
                depthWriteEnabled: true,
                depthCompare: "less",
            },
        });

        this.uniformsManager = uniformsManager;
        this.staticMeshManager = staticMeshManager;
        this.rasterizeStorageBindGroup = rasterizeStorageBindGroup;

        this.indexCount = staticMeshManager.numIndices;
    }

    addDraw(renderPassEncoder: GPURenderPassEncoder) {
        renderPassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        renderPassEncoder.setBindGroup(1, this.rasterizeStorageBindGroup);
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.setVertexBuffer(0, this.staticMeshManager.verticesBuffer);
        renderPassEncoder.setIndexBuffer(this.staticMeshManager.indicesBuffer, "uint32");
        renderPassEncoder.drawIndexed(this.indexCount);
    }
}