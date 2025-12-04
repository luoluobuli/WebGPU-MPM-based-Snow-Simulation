import type { GpuUniformsBufferManager } from "$lib/gpu/uniforms/GpuUniformsBufferManager";
import ssfrDepthImpostorVertSrc from "./ssfrDepthImpostor.vert.wgsl?raw";
import ssfrDepthImpostorFragSrc from "./ssfrDepthImpostor.frag.wgsl?raw";
import type { GpuMpmBufferManager } from "../mpm/GpuMpmBufferManager";
import type { GpuRenderMethod } from "$lib/gpu/GpuRenderMethod";
import { attachPrelude } from "$lib/gpu/shaderPrelude";

export class GpuSsfrRenderPipelineManager implements GpuRenderMethod {
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
            label: "ssfr render pipeline bind group layout",
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
            label: "ssfr render pipeline bind group",
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
            label: "ssfr depth impostor vertex module",
            code: attachPrelude(ssfrDepthImpostorVertSrc),
        });
        const fragmentModule = device.createShaderModule({
            label: "ssfr depth impostor fragment module",
            code: attachPrelude(ssfrDepthImpostorFragSrc),
        });
        
        const renderPipelineLayout = device.createPipelineLayout({
            label: "ssfr render pipeline layout",
            bindGroupLayouts: [bindGroupLayout],
        });
        this.renderPipeline = device.createRenderPipeline({
            label: "ssfr render pipeline",

            layout: renderPipelineLayout,

            vertex: {
                module: vertexModule,
                entryPoint: "vert",
            },

            fragment: {
                module: fragmentModule,
                entryPoint: "frag",
                targets: [
                    {
                        format,
                        // writeMask: 0, // Removed to allow color output for debugging
                    },
                ],
            },

            primitive: {
                topology: "triangle-list",
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

    addDraw(renderPassEncoder: GPURenderPassEncoder) {
        renderPassEncoder.setBindGroup(0, this.bindGroup);
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.draw(6, this.mpmManager.nParticles, 0, 0);
    }
}
