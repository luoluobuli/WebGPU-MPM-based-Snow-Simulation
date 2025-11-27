import { attachPrelude } from "../shaderPrelude";
import type { GpuUniformsBufferManager } from "../uniforms/GpuUniformsBufferManager";
import type { GpuVolumetricBufferManager } from "./GpuVolumetricBufferManager";
import type { GpuMpmBufferManager } from "../mpm/GpuMpmBufferManager";
import calculateGridDensitySrc from "./calculateGridDensity.wgsl?raw";
import volumetricRaymarchSrc from "./volumetricRaymarch.cs.wgsl?raw";
import volumetricVertexSrc from "./volumetricVertex.wgsl?raw";
import volumetricFragmentSrc from "./volumetricFragment.wgsl?raw";
import preludeSrc from "./prelude.wgsl?raw";

export class GpuVolumetricRenderPipelineManager {
    readonly densityRasterizePipeline: GPUComputePipeline;
    readonly volumetricRaymarchPipeline: GPUComputePipeline;
    readonly renderPipeline: GPURenderPipeline;
    
    densityBindGroup: GPUBindGroup;
    raymarchBindGroup: GPUBindGroup;
    renderBindGroup: GPUBindGroup;

    readonly vertBuffer: GPUBuffer;

    private readonly uniformsManager: GpuUniformsBufferManager;
    private readonly volumetricBufferManager: GpuVolumetricBufferManager;
    private readonly raymarchBindGroupLayout: GPUBindGroupLayout;

    constructor({
        device,
        format,
        uniformsManager,
        volumetricBufferManager,
        mpmBufferManager,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        uniformsManager: GpuUniformsBufferManager,
        volumetricBufferManager: GpuVolumetricBufferManager,
        mpmBufferManager: GpuMpmBufferManager,
    }) {
        this.uniformsManager = uniformsManager;
        this.volumetricBufferManager = volumetricBufferManager;



        const densityBindGroupLayout = device.createBindGroupLayout({
            label: "volumetric density bind group layout",
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

        this.densityBindGroup = device.createBindGroup({
            label: "volumetric density bind group",
            layout: densityBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: mpmBufferManager.particleDataBuffer },
                },
                {
                    binding: 1,
                    resource: { buffer: volumetricBufferManager.densityGridBuffer },
                },
            ],
        });



        const densityPipelineLayout = device.createPipelineLayout({
            label: "volumetric density pipeline layout",
            bindGroupLayouts: [uniformsManager.bindGroupLayout, densityBindGroupLayout],
        });

        this.densityRasterizePipeline = device.createComputePipeline({
            label: "volumetric density rasterize pipeline",
            layout: densityPipelineLayout,
            compute: {
                module: device.createShaderModule({
                    code: attachPrelude(`${preludeSrc}\n${calculateGridDensitySrc}`),
                }),
                entryPoint: "calculateGridDensity",
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
            ],
        });
        this.raymarchBindGroupLayout = raymarchBindGroupLayout;

        this.raymarchBindGroup = device.createBindGroup({
            label: "volumetric raymarch bind group",
            layout: raymarchBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: volumetricBufferManager.densityGridBuffer },
                },
                {
                    binding: 1,
                    resource: this.volumetricBufferManager.outputTextureView,
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
                    resource: this.volumetricBufferManager.outputTextureView,
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
                module: device.createShaderModule({ code: `${preludeSrc}\n${volumetricVertexSrc}` }),
                entryPoint: "vert",
                buffers: [
                    {
                        arrayStride: 8,
                        attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
                    },
                ],
            },
            fragment: {
                module: device.createShaderModule({ code: `${preludeSrc}\n${volumetricFragmentSrc}` }),
                entryPoint: "frag",
                targets: [
                    {
                        format,
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

    addDensityDispatch(computePassEncoder: GPUComputePassEncoder, nParticles: number) {
        computePassEncoder.setPipeline(this.densityRasterizePipeline);
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.densityBindGroup);
        computePassEncoder.dispatchWorkgroups(Math.ceil(nParticles / 256));
    }

    addRaymarchDispatch(computePassEncoder: GPUComputePassEncoder) {
        computePassEncoder.setPipeline(this.volumetricRaymarchPipeline);
        computePassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        computePassEncoder.setBindGroup(1, this.raymarchBindGroup);
        
        const { width, height } = this.volumetricBufferManager.outputTexture;
        computePassEncoder.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
    }

    addDraw(renderPassEncoder: GPURenderPassEncoder) {
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.setBindGroup(0, this.renderBindGroup);
        renderPassEncoder.setVertexBuffer(0, this.vertBuffer);
        renderPassEncoder.draw(4);
    }

    resize(device: GPUDevice, width: number, height: number) {
        this.volumetricBufferManager.resize(device, width, height);

        this.raymarchBindGroup = device.createBindGroup({
            label: "volumetric raymarch bind group",
            layout: this.raymarchBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.volumetricBufferManager.densityGridBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: this.volumetricBufferManager.outputTextureView,
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
                    resource: this.volumetricBufferManager.outputTextureView,
                },
            ],
        });
    }
}
