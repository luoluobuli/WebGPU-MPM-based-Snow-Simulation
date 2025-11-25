import cuboidVertexModuleSrc from "./cuboidVertex.wgsl?raw";
import cuboidFragmentModuleSrc from "./cuboidFragment.wgsl?raw";
import preludeSrc from "./prelude.wgsl?raw";
import { attachPrelude } from "../shaderPrelude";
import type { GpuRenderMethod } from "$lib/gpu/GpuRenderMethod";
import type { GpuUniformsBufferManager } from "../uniforms/GpuUniformsBufferManager";
import type { GpuMpmBufferManager } from "../mpm/GpuMpmBufferManager";

export class GpuMpmGridRenderPipelineManager implements GpuRenderMethod {
    readonly renderPipeline: GPURenderPipeline;
    readonly uniformsManager: GpuUniformsBufferManager;
    readonly mpmManager: GpuMpmBufferManager;
    readonly linesBuffer: GPUBuffer;
    
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
        const vertexModule = device.createShaderModule({
            label: "mpm grid vertex module",
            code: attachPrelude(`${preludeSrc}${cuboidVertexModuleSrc}`),
        });
        const fragmentModule = device.createShaderModule({
            label: "mpm grid fragment module",
            code: attachPrelude(`${preludeSrc}${cuboidFragmentModuleSrc}`),
        });

        this.linesBuffer = device.createBuffer({
            label: "mpm grid cuboid lines buffer",
            size: 24 * 4,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.linesBuffer, 0, new Uint32Array([
            0, 1,
            1, 2,
            2, 3,
            3, 0,

            4, 5,
            5, 6,
            6, 7,
            7, 4,

            0, 4,
            1, 5,
            2, 6,
            3, 7,
        ]));
        
        
        const renderPipelineLayout = device.createPipelineLayout({
            label: "mpm grid render pipeline layout",
            bindGroupLayouts: [
                uniformsManager.bindGroupLayout,
            ],
        });
        this.renderPipeline = device.createRenderPipeline({
            label: "mpm grid render pipeline",
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
                                format: "uint32",
                            },
                        ],
                        arrayStride: 4,
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
                topology: "line-list",
            },

            depthStencil: {
                format: depthFormat,
                depthWriteEnabled: true,
                depthCompare: "less",
            },
        });

        this.uniformsManager = uniformsManager;
        this.mpmManager = mpmManager;
    }

    addDraw(renderPassEncoder: GPURenderPassEncoder) {
        renderPassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.setVertexBuffer(0, this.linesBuffer);
        renderPassEncoder.draw(24);
    }
}