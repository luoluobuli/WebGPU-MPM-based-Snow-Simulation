import { attachPrelude } from "../shaderPrelude";
import type { GpuUniformsBufferManager } from "../uniforms/GpuUniformsBufferManager";
import type { GpuEnvironmentTextureManager } from "./GpuEnvironmentTextureManager";
import environmentFragmentSrc from "./environmentFragment.wgsl?raw";
import environmentVertexSrc from "./environmentVertex.wgsl?raw";
import preludeSrc from "./prelude.wgsl?raw";


export class GpuEnvironmentRenderPipelineManager {
    readonly renderPipeline: GPURenderPipeline;

    private readonly bindGroup: GPUBindGroup;
    private readonly vertBuffer: GPUBuffer;

    constructor({
        device,
        textureManager,
        uniformsManager,
        format,
    }: {
        device: GPUDevice,
        textureManager: GpuEnvironmentTextureManager,
        uniformsManager: GpuUniformsBufferManager,
        format: GPUTextureFormat,
    }) {
        const textureBindGroupLayout = device.createBindGroupLayout({
            label: "environment texture bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: "uniform",
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: {
                        sampleType: "float",
                    },
                },
            ],
        });

        const bindGroup = device.createBindGroup({
            label: "environment texture bind group",
            layout: textureBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: uniformsManager.buffer,
                    },
                },
                {
                    binding: 1,
                    resource: textureManager.environmentTexture.createView(),
                },
            ],
        });



        const renderPipelineLayout = device.createPipelineLayout({
            label: "environment texture render pipeline layout",
            bindGroupLayouts: [
                textureBindGroupLayout,
            ],
        });

        this.renderPipeline = device.createRenderPipeline({
            label: "environment texture render pipeline",
            layout: renderPipelineLayout,
            vertex: {
                module: device.createShaderModule({
                    label: "environment texture vertex shader module",
                    code: attachPrelude(`${preludeSrc}\n${environmentVertexSrc}`),
                }),
                entryPoint: "vert",
                buffers: [
                    {
                        arrayStride: 8,
                        attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
                    },
                ],
            },
            fragment: {
                module: device.createShaderModule({
                    label: "environment texture fragment shader module",
                    code: attachPrelude(`${preludeSrc}\n${environmentFragmentSrc}`),
                }),
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

            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: "depth24plus",
            },
        });

        this.bindGroup = bindGroup;

        this.vertBuffer = device.createBuffer({
            label: "environment texture vertex buffer",
            size: 32,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.vertBuffer, 0, new Float32Array([
            -1, -1,
            -1, 1,
            1, -1,
            1, 1,
        ]));
    }


    addDraw(renderPassEncoder: GPURenderPassEncoder) {
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.setBindGroup(0, this.bindGroup);
        renderPassEncoder.setVertexBuffer(0, this.vertBuffer);
        renderPassEncoder.draw(4);
    }
}