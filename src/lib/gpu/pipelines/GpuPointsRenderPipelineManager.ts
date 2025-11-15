import type { GpuUniformsBufferManager } from "$lib/gpu/buffers/GpuUniformsBufferManager";
import commonModuleSrc from "$lib/gpu/shaders/_common.wgsl?raw";
import pointsVertexModuleSrc from "$lib/gpu/shaders/pointsVertex.wgsl?raw";
import pointsFragmentModuleSrc from "$lib/gpu/shaders/pointsFragment.wgsl?raw";

export class GpuPointsRenderPipelineManager {
    readonly renderPipeline: GPURenderPipeline;

    readonly uniformsManager: GpuUniformsBufferManager;

    constructor({
        device,
        format,
        uniformsManager,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        uniformsManager: GpuUniformsBufferManager,
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
                        arrayStride: 48,
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
    }

    addRenderPass({
        commandEncoder,
        context,
        particleDataBuffer,
        nParticles,
    }: {
        commandEncoder: GPUCommandEncoder,
        context: GPUCanvasContext,
        particleDataBuffer: GPUBuffer,
        nParticles: number,
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
        renderPassEncoder.setVertexBuffer(0, particleDataBuffer);
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.draw(nParticles);
        renderPassEncoder.end();
    }
}