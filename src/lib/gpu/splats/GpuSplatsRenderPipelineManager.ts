import type { GpuUniformsBufferManager } from "$lib/gpu/uniforms/GpuUniformsBufferManager";
import splatsVertexModuleSrc from "./splatsRender.vert.wgsl?raw";
import splatsFragmentModuleSrc from "./splatsRender.frag.wgsl?raw";
import type { GpuMpmBufferManager } from "../mpm/GpuMpmBufferManager";
import type { GpuRenderMethod } from "$lib/gpu/GpuRenderMethod";
import { attachPrelude } from "$lib/gpu/shaderPrelude";
import particleAppearanceSrc from "../particleAppearance/particleAppearance.wgsl?raw";
import type { GpuParticleAppearanceBufferManager } from "../particleAppearance/GpuParticleAppearanceBufferManager";

const prerenderPasses: string[] = [];

export class GpuSplatsRenderPipelineManager implements GpuRenderMethod {
    private readonly renderPipeline: GPURenderPipeline;
    private readonly bindGroup: GPUBindGroup;
    private readonly mpmManager: GpuMpmBufferManager;

    constructor({
        device,
        format,
        depthFormat,
        uniformsManager,
        mpmManager,
        particleAppearanceManager,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        depthFormat: GPUTextureFormat,
        uniformsManager: GpuUniformsBufferManager,
        mpmManager: GpuMpmBufferManager,
        particleAppearanceManager: GpuParticleAppearanceBufferManager,
    }) {
        const bindGroupLayout = device.createBindGroupLayout({
            label: "splats render pipeline bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: "uniform",
                    },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: {
                        type: "read-only-storage",
                    },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: {
                        type: "read-only-storage",
                    },
                },
            ],
        });

        this.bindGroup = device.createBindGroup({
            label: "splats render pipeline bind group",
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: uniformsManager.buffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: mpmManager.particleDataBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: particleAppearanceManager.appearanceBuffer,
                    },
                },
            ],
        });

        const vertexModule = device.createShaderModule({
            label: "splats vertex module",
            code: attachPrelude(`${particleAppearanceSrc}${splatsVertexModuleSrc}`),
        });
        const fragmentModule = device.createShaderModule({
            label: "splats fragment module",
            code: attachPrelude(splatsFragmentModuleSrc),
        });

        this.renderPipeline = device.createRenderPipeline({
            label: "splats render pipeline",
            layout: device.createPipelineLayout({
                label: "splats render pipeline layout",
                bindGroupLayouts: [bindGroupLayout],
            }),
            vertex: {
                module: vertexModule,
                entryPoint: "vert",
            },
            fragment: {
                module: fragmentModule,
                entryPoint: "frag",
                targets: [{ format }],
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

        this.mpmManager = mpmManager;
    }

    prerenderPasses(): string[] {
        return prerenderPasses;
    }

    addPrerenderPasses(_commandEncoder: GPUCommandEncoder, _depthTextureView: GPUTextureView) {}

    addCompositeDraw(renderPassEncoder: GPURenderPassEncoder) {
        renderPassEncoder.setBindGroup(0, this.bindGroup);
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.draw(6, this.mpmManager.nParticles, 0, 0);
    }

    resize(_device: GPUDevice, _width: number, _height: number, _depthTextureView: GPUTextureView): void {}

    destroy() {}
}
