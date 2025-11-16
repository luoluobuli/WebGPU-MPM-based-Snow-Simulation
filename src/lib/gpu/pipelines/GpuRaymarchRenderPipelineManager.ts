import type { GpuUniformsBufferManager } from "$lib/gpu/buffers/GpuUniformsBufferManager";
import commonModuleSrc from "$lib/gpu/shaders/_common.wgsl?raw";
import raymarchVertexModuleSrc from "$lib/gpu/shaders/raymarchVertex.wgsl?raw";
import raymarchFragmentModuleSrc from "$lib/gpu/shaders/raymarchFragment.wgsl?raw";
import type { GpuMpmBufferManager } from "../buffers/GpuMpmBufferManager";

export class GpuRaymarchRenderPipelineManager {
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
            code: commonModuleSrc + raymarchVertexModuleSrc,
        });
        const fragmentModule = device.createShaderModule({
            label: "raymarch fragment module",
            code: commonModuleSrc + raymarchFragmentModuleSrc,
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

    addRenderPass({
        commandEncoder,
        context,
    }: {
        commandEncoder: GPUCommandEncoder,
        context: GPUCanvasContext,
    }) {
        const renderPassEncoder = commandEncoder.beginRenderPass({
            label: "points render pass",
            colorAttachments: [
                {
                    clearValue: {
                        r: 0,
                        g: 0,
                        b: 0,
                        a: 1,
                    },

                    loadOp: "clear",
                    storeOp: "store",
                    view: context.getCurrentTexture().createView(),
                },
            ],
        });
        renderPassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        renderPassEncoder.setBindGroup(1, this.raymarchStorageBindGroup);
        renderPassEncoder.setVertexBuffer(0, this.fullscreenBuffer);
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.draw(4);
        renderPassEncoder.end();
    }
}