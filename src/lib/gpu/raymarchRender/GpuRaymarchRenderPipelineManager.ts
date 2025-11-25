import type { GpuUniformsBufferManager } from "$lib/gpu/uniforms/GpuUniformsBufferManager";
import raymarchVertexModuleSrc from "./raymarchVertex.wgsl?raw";
import raymarchFragmentModuleSrc from "./raymarchFragment.wgsl?raw";
import type { GpuMpmBufferManager } from "../mpm/GpuMpmBufferManager";
import type { GpuRenderMethod } from "$lib/gpu/GpuRenderMethod";
import { attachPrelude } from "$lib/gpu/shaderPrelude";
import preludeSrc from "./prelude.wgsl?raw";

export class GpuRaymarchRenderPipelineManager implements GpuRenderMethod {
    readonly renderPipeline: GPURenderPipeline;
    readonly raymarchStorageBindGroup: GPUBindGroup;
    readonly fullscreenBuffer: GPUBuffer;

    readonly uniformsManager: GpuUniformsBufferManager;


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
        const raymarchStorageBindGroupLayout = device.createBindGroupLayout({
            label: "raymarch storage bind group layout",
            
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: "read-only-storage",
                    },
                },
            ],
        });
        
        const raymarchStorageBindGroup = device.createBindGroup({
            label: "raymarch step storage bind group",

            layout: raymarchStorageBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: mpmManager.particleDataBuffer,
                    },
                },
            ],
        });


        const vertexModule = device.createShaderModule({
            label: "raymarch vertex module",
            code: attachPrelude(`${preludeSrc}${raymarchVertexModuleSrc}`),
        });
        const fragmentModule = device.createShaderModule({
            label: "raymarch fragment module",
            code: attachPrelude(`${preludeSrc}${raymarchFragmentModuleSrc}`),
        });
        
        const renderPipelineLayout = device.createPipelineLayout({
            label: "raymarch render pipeline",
            bindGroupLayouts: [
                uniformsManager.bindGroupLayout,
                raymarchStorageBindGroupLayout,
            ],
        });

        const fullscreenBuffer = device.createBuffer({
            label: "raymarch fullscreen vertex buffer",
            size: 32,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(fullscreenBuffer, 0, new Float32Array([
            -1, -1,
            -1, 1,
            1, -1,
            1, 1,
        ]));

        this.fullscreenBuffer = fullscreenBuffer;
        this.renderPipeline = device.createRenderPipeline({
            label: "raymarch render pipeline",

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
                                format: "float32x2",
                            },
                        ],
                        arrayStride: 8,
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
                topology: "triangle-strip",
            },
        });

        this.uniformsManager = uniformsManager;
        this.raymarchStorageBindGroup = raymarchStorageBindGroup;
    }

    addDraw(renderPassEncoder: GPURenderPassEncoder) {
        renderPassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        renderPassEncoder.setBindGroup(1, this.raymarchStorageBindGroup);
        renderPassEncoder.setVertexBuffer(0, this.fullscreenBuffer);
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.draw(4);
    }
}