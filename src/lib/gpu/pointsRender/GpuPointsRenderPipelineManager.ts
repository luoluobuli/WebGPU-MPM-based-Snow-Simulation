import type { GpuUniformsBufferManager } from "$lib/gpu/uniforms/GpuUniformsBufferManager";
import pointsVertexModuleSrc from "./pointsVertex.wgsl?raw";
import pointsFragmentModuleSrc from "./pointsFragment.wgsl?raw";
import type { GpuMpmBufferManager } from "../mpm/GpuMpmBufferManager";
import type { GpuRenderMethod } from "$lib/gpu/GpuRenderMethod";
import { attachPrelude } from "$lib/gpu/shaderPrelude";
import preludeSrc from "./prelude.wgsl?raw";

const prerenderPasses: string[] = [];

export class GpuPointsRenderPipelineManager implements GpuRenderMethod {
    readonly renderPipeline: GPURenderPipeline;

    readonly uniformsManager: GpuUniformsBufferManager;
    readonly mpmManager: GpuMpmBufferManager;

    private readonly bindGroup: GPUBindGroup;

    constructor({
        device,
        format,
        depthFormat,
        uniformsManager,
        mpmManager,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        depthFormat: GPUTextureFormat,
        uniformsManager: GpuUniformsBufferManager,
        mpmManager: GpuMpmBufferManager,
    }) {
        const bindGroupLayout = device.createBindGroupLayout({
            label: "points render pipeline bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: "uniform",
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: {
                        type: "read-only-storage",
                    },
                },
            ],
        });

        const bindGroup = device.createBindGroup({
            label: "points render pipeline bind group",
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: uniformsManager.buffer,
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: mpmManager.particleDataBuffer,
                    },
                },
            ],
        });


        const vertexModule = device.createShaderModule({
            label: "points vertex module",
            code: attachPrelude(`${preludeSrc}${pointsVertexModuleSrc}`),
        });
        const fragmentModule = device.createShaderModule({
            label: "points fragment module",
            code: attachPrelude(`${preludeSrc}${pointsFragmentModuleSrc}`),
        });
        
        const renderPipelineLayout = device.createPipelineLayout({
            label: "points render pipeline",
            bindGroupLayouts: [bindGroupLayout],
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
                        arrayStride: 192,
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

            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: depthFormat,
            },
        });

        this.uniformsManager = uniformsManager;
        this.mpmManager = mpmManager;

        this.bindGroup = bindGroup;
    }

    prerenderPasses(): string[] {
        return prerenderPasses;
    }
    
    addPrerenderPasses(commandEncoder: GPUCommandEncoder, depthTextureView: GPUTextureView) {}

    addFinalDraw(renderPassEncoder: GPURenderPassEncoder) {
        renderPassEncoder.setBindGroup(0, this.bindGroup);
        renderPassEncoder.setVertexBuffer(0, this.mpmManager.particleDataBuffer);
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.draw(this.mpmManager.nParticles);
    }

    resize(device: GPUDevice, width: number, height: number, depthTextureView: GPUTextureView): void {}

    destroy() {}
}