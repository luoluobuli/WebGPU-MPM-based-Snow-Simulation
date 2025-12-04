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
                        writeMask: 0, // We only care about depth for now? Or do we want to output color?
                        // The task is "Depth Impostor". Usually this writes to depth buffer.
                        // But we are in a render pass that has a color attachment.
                        // If we don't write color, we should set writeMask to 0.
                        // However, for debugging, maybe we want to output something?
                        // Let's output depth as color or just white?
                        // The fragment shader returns `@builtin(frag_depth)`. It doesn't return color.
                        // So we MUST set writeMask to 0 or change fragment shader to return color.
                        // If we set writeMask to 0, we don't need to return color.
                        // But the pipeline expects a target if the render pass has one.
                        // The render pass in GpuSnowPipelineRunner has a color attachment.
                        // So we must provide a target state.
                        // writeMask: 0 is correct if we don't want to write color.
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
