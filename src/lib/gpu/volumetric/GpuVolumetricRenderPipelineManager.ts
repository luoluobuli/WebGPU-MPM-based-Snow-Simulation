import { attachPrelude } from "../shaderPrelude";
import type { GpuUniformsBufferManager } from "../uniforms/GpuUniformsBufferManager";
import type { GpuVolumetricBufferManager } from "./GpuVolumetricBufferManager";
import type { GpuMpmBufferManager } from "../mpm/GpuMpmBufferManager";
import type { GpuEnvironmentTextureManager } from "../environmentMap/GpuEnvironmentTextureManager";
import calculateGridMassSrc from "./calculateGridMass.wgsl?raw";
import volumetricRaymarchSrc from "./volumetricRaymarch.cs.wgsl?raw";
import volumetricVertexSrc from "./volumetricVertex.wgsl?raw";
import volumetricFragmentSrc from "./volumetricFragment.wgsl?raw";
import preludeSrc from "./prelude.wgsl?raw";
import type { GpuRenderMethod } from "../GpuRenderMethod";

export class GpuVolumetricRenderPipelineManager implements GpuRenderMethod {
    readonly massCalulationPipeline: GPUComputePipeline;
    readonly volumetricRaymarchPipeline: GPUComputePipeline;
    readonly renderPipeline: GPURenderPipeline;
    
    massCalulationBindGroup: GPUBindGroup;
    raymarchBindGroup: GPUBindGroup;
    renderBindGroup: GPUBindGroup;

    readonly vertBuffer: GPUBuffer;

    private readonly mpmManager: GpuMpmBufferManager;
    private readonly uniformsManager: GpuUniformsBufferManager;
    private readonly volumetricBufferManager: GpuVolumetricBufferManager;
    private readonly environmentTextureManager: GpuEnvironmentTextureManager;
    private readonly raymarchBindGroupLayout: GPUBindGroupLayout;
    private readonly sampler: GPUSampler;

    constructor({
        device,
        format,
        uniformsManager,
        volumetricBufferManager,
        mpmBufferManager,
        environmentTextureManager,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        uniformsManager: GpuUniformsBufferManager,
        volumetricBufferManager: GpuVolumetricBufferManager,
        mpmBufferManager: GpuMpmBufferManager,
        environmentTextureManager: GpuEnvironmentTextureManager,
    }) {
        this.uniformsManager = uniformsManager;
        this.volumetricBufferManager = volumetricBufferManager;
        this.environmentTextureManager = environmentTextureManager;
        this.mpmManager = mpmBufferManager;

        this.sampler = device.createSampler({
            magFilter: "linear",
            minFilter: "linear",
        });



        const massCalulationBindGroupLayout = device.createBindGroupLayout({
            label: "volumetric mass calculation bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
            ],
        });

        this.massCalulationBindGroup = device.createBindGroup({
            label: "volumetric mass calculation bind group",
            layout: massCalulationBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: mpmBufferManager.particleDataBuffer },
                },
                {
                    binding: 1,
                    resource: { buffer: volumetricBufferManager.massGridBuffer },
                },
            ],
        });

        this.massCalulationBindGroup = device.createBindGroup({
            label: "volumetric mass calculation bind group",
            layout: massCalulationBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: mpmBufferManager.particleDataBuffer },
                },
                {
                    binding: 1,
                    resource: { buffer: volumetricBufferManager.massGridBuffer },
                },
            ],
        });



        const massCalulationPipelineLayout = device.createPipelineLayout({
            label: "volumetric mass calculation pipeline layout",
            bindGroupLayouts: [uniformsManager.bindGroupLayout, massCalulationBindGroupLayout],
        });

        this.massCalulationPipeline = device.createComputePipeline({
            label: "volumetric mass calculation pipeline",
            layout: massCalulationPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    code: attachPrelude(`${preludeSrc}\n${calculateGridMassSrc}`),
                }),
                entryPoint: "calculateGridMass",
            },
        });




        const raymarchBindGroupLayout = device.createBindGroupLayout({
            label: "volumetric raymarch bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        access: "write-only",
                        format: "rgba8unorm",
                        viewDimension: "2d",
                },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                storageTexture: {
                    access: "write-only",
                    format: "r32float",
                    viewDimension: "2d",
                },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                texture: { sampleType: "float" },
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                sampler: { type: "filtering" },
            },
        ],
        });
        this.raymarchBindGroupLayout = raymarchBindGroupLayout;

        this.raymarchBindGroup = device.createBindGroup({
            label: "volumetric raymarch bind group",
            layout: raymarchBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: volumetricBufferManager.massGridBuffer },
                },
                {
                    binding: 1,
                    resource: this.volumetricBufferManager.outputTextureView,
                },
                {
                    binding: 2,
                    resource: this.volumetricBufferManager.depthTextureView,
                },
                {
                    binding: 3,
                    resource: this.environmentTextureManager.environmentTexture.createView(),
                },
                {
                    binding: 4,
                    resource: this.sampler,
                },
            ],
        });



        const raymarchPipelineLayout = device.createPipelineLayout({
            label: "volumetric raymarch pipeline layout",
            bindGroupLayouts: [
                uniformsManager.bindGroupLayout,
                raymarchBindGroupLayout,
            ],
        });

        this.volumetricRaymarchPipeline = device.createComputePipeline({
            label: "volumetric raymarch pipeline",
            layout: raymarchPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    code: attachPrelude(`${preludeSrc}\n${volumetricRaymarchSrc}`),
                }),
                entryPoint: "doVolumetricRaymarch",
            },
        });




        const renderBindGroupLayout = device.createBindGroupLayout({
            label: "volumetric render bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "unfilterable-float" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "unfilterable-float" },
                },
            ],
        });

        this.renderBindGroup = device.createBindGroup({
            label: "volumetric render bind group",
            layout: renderBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.uniformsManager.buffer,
                },
                {
                    binding: 1,
                    resource: this.volumetricBufferManager.outputTextureView,
                },
                {
                    binding: 2,
                    resource: this.volumetricBufferManager.depthTextureView,
                },
            ],
        });

        const renderPipelineLayout = device.createPipelineLayout({
            label: "volumetric render pipeline layout",
            bindGroupLayouts: [renderBindGroupLayout],
        });

        this.renderPipeline = device.createRenderPipeline({
            label: "volumetric render pipeline",
            layout: renderPipelineLayout,
            vertex: {
                module: device.createShaderModule({ code: attachPrelude(`${preludeSrc}\n${volumetricVertexSrc}`) }),
                entryPoint: "vert",
                buffers: [
                    {
                        arrayStride: 8,
                        attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
                    },
                ],
            },
            fragment: {
                module: device.createShaderModule({ code: attachPrelude(`${preludeSrc}\n${volumetricFragmentSrc}`) }),
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
            primitive: { topology: "triangle-strip" },

            depthStencil: {
                depthWriteEnabled: false,
                depthCompare: "less",
                format: "depth24plus",
            },
        });

        this.vertBuffer = device.createBuffer({
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

    addMassCalulationDispatch(computePassEncoder: GPUComputePassEncoder, nParticles: number) {
        computePassEncoder.setPipeline(this.massCalulationPipeline);
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.massCalulationBindGroup);
        computePassEncoder.dispatchWorkgroups(Math.ceil(nParticles / 256));
    }

    addRaymarchDispatch(computePassEncoder: GPUComputePassEncoder) {
        computePassEncoder.setPipeline(this.volumetricRaymarchPipeline);
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.raymarchBindGroup);
        
        const { width, height } = this.volumetricBufferManager.outputTexture;
        computePassEncoder.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
    }

    nPrerenderPasses(): number {
        return 1;
    }

    addPrerenderPasses(commandEncoder: GPUCommandEncoder, depthTextureView: GPUTextureView) {
        commandEncoder.clearBuffer(this.volumetricBufferManager.massGridBuffer);

        const prerenderComputePassEncoder = commandEncoder.beginComputePass({
            label: "volumetric compute pass",
            // timestampWrites: this.performanceMeasurementManager !== null
            //     ? {
            //         querySet: this.performanceMeasurementManager.querySet,
            //         beginningOfPassWriteIndex: 2,
            //         endOfPassWriteIndex: 3,
            //     }
            //     : undefined,
        });

        this.addMassCalulationDispatch(prerenderComputePassEncoder, this.mpmManager.nParticles);
        this.addRaymarchDispatch(prerenderComputePassEncoder);
        
        prerenderComputePassEncoder.end();
    }

    addFinalDraw(renderPassEncoder: GPURenderPassEncoder) {
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.setBindGroup(0, this.renderBindGroup);
        renderPassEncoder.setVertexBuffer(0, this.vertBuffer);
        renderPassEncoder.draw(4);
    }

    resize(device: GPUDevice, width: number, height: number, depthTextureView: GPUTextureView) {
        this.volumetricBufferManager.resize(device, width, height);

        this.raymarchBindGroup = device.createBindGroup({
            label: "volumetric raymarch bind group",
            layout: this.raymarchBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.volumetricBufferManager.massGridBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: this.volumetricBufferManager.outputTextureView,
                },
                {
                    binding: 2,
                    resource: this.volumetricBufferManager.depthTextureView,
                },
                {
                    binding: 3,
                    resource: this.environmentTextureManager.environmentTexture.createView(),
                },
                {
                    binding: 4,
                    resource: this.sampler,
                },
            ],
        });

        const renderBindGroupLayout = this.renderPipeline.getBindGroupLayout(0);
        this.renderBindGroup = device.createBindGroup({
            label: "volumetric render bind group",
            layout: renderBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.uniformsManager.buffer,
                },
                {
                    binding: 1,
                    resource: this.volumetricBufferManager.outputTextureView,
                },
                {
                    binding: 2,
                    resource: this.volumetricBufferManager.depthTextureView,
                },
            ],
        });
    }

    destroy() {
        this.vertBuffer.destroy();

        this.volumetricBufferManager.destroy();
    }
}
