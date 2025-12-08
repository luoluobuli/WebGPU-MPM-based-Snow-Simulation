
import { attachPrelude } from "../shaderPrelude";
import ssaoVertSrc from "./ssao.vert.wgsl?raw";
import ssaoFragSrc from "./ssao.frag.wgsl?raw";
import type { GpuUniformsBufferManager } from "../uniforms/GpuUniformsBufferManager";

export class GpuSsaoPipelineManager {
    private readonly device: GPUDevice;
    private readonly pipeline: GPURenderPipeline;
    private readonly uniformsManager: GpuUniformsBufferManager;
    private bindGroupLayout: GPUBindGroupLayout;
    private bindGroup: GPUBindGroup | null = null;

    constructor({
        device,
        format,
        uniformsManager,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        uniformsManager: GpuUniformsBufferManager,
    }) {
        this.device = device;
        this.uniformsManager = uniformsManager;

        this.bindGroupLayout = device.createBindGroupLayout({
            label: "SSAO bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "depth" },
                },
            ],
        });

        this.pipeline = device.createRenderPipeline({
            label: "SSAO pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [this.bindGroupLayout],
            }),
            vertex: {
                module: device.createShaderModule({
                    code: ssaoVertSrc,
                }),
                entryPoint: "vert",
            },
            fragment: {
                module: device.createShaderModule({
                    code: attachPrelude(ssaoFragSrc),
                }),
                entryPoint: "frag",
                targets: [
                    {
                        format,
                        blend: {
                            color: {
                                srcFactor: "one",
                                dstFactor: "one-minus-src-alpha",
                            },
                            alpha: {
                                srcFactor: "one",
                                dstFactor: "one-minus-src-alpha",
                            },
                        },
                    },
                ],
            },
            primitive: {
                topology: "triangle-strip",
            },
        });
    }

    resize(depthTextureView: GPUTextureView) {
        this.bindGroup = this.device.createBindGroup({
            label: "SSAO bind group",
            layout: this.bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.uniformsManager.buffer },
                },
                {
                    binding: 1,
                    resource: depthTextureView,
                },
            ],
        });
    }

    addDraw(renderPassEncoder: GPURenderPassEncoder) {
        if (!this.bindGroup) {
            console.warn("SSAO bind group not created. Call resize() first.");
            return;
        }
        renderPassEncoder.setPipeline(this.pipeline);
        renderPassEncoder.setBindGroup(0, this.bindGroup);
        renderPassEncoder.draw(4);
    }
}
