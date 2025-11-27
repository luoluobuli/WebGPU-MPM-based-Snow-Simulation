import { attachPrelude } from "../shaderPrelude";
import type { GpuUniformsBufferManager } from "../uniforms/GpuUniformsBufferManager";
import type { GpuVolumetricBufferManager } from "./GpuVolumetricBufferManager";
import type { GpuMpmBufferManager } from "../mpm/GpuMpmBufferManager";
import densityRasterizeSrc from "./densityRasterize.cs.wgsl?raw";
import volumetricRaymarchSrc from "./volumetricRaymarch.cs.wgsl?raw";
import blitSrc from "./blit.wgsl?raw";

export class GpuVolumetricRenderPipelineManager {
    readonly densityRasterizePipeline: GPUComputePipeline;
    readonly volumetricRaymarchPipeline: GPUComputePipeline;
    readonly blitPipeline: GPURenderPipeline;
    
    densityBindGroup: GPUBindGroup;
    raymarchBindGroup: GPUBindGroup;
    blitBindGroup: GPUBindGroup;

    readonly fullscreenBuffer: GPUBuffer;

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

        // 1. Density Rasterize Pipeline
        const densityBindGroupLayout = device.createBindGroupLayout({
            label: "volumetric density bind group layout",
            entries: [
                {
                    binding: 0, // Particle Data (Read)
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 1, // Density Grid (Read-Write)
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
                    code: attachPrelude(densityRasterizeSrc),
                }),
                entryPoint: "doDensityRasterize",
            },
        });

        // 2. Volumetric Raymarch Pipeline
        const raymarchBindGroupLayout = device.createBindGroupLayout({
            label: "volumetric raymarch bind group layout",
            entries: [
                {
                    binding: 0, // Density Grid (Read)
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
                {
                    binding: 1, // Output Texture (Write)
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
                    code: attachPrelude(volumetricRaymarchSrc),
                }),
                entryPoint: "doVolumetricRaymarch",
            },
        });

        // 3. Blit Pipeline
        const blitBindGroupLayout = device.createBindGroupLayout({
            label: "volumetric blit bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "unfilterable-float" }, // storage texture is not filterable usually, but we are binding as texture_2d<f32>
                },
            ],
        });

        this.blitBindGroup = device.createBindGroup({
            label: "volumetric blit bind group",
            layout: blitBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.volumetricBufferManager.outputTextureView,
                },
            ],
        });

        const blitPipelineLayout = device.createPipelineLayout({
            label: "volumetric blit pipeline layout",
            bindGroupLayouts: [blitBindGroupLayout],
        });

        this.blitPipeline = device.createRenderPipeline({
            label: "volumetric blit pipeline",
            layout: blitPipelineLayout,
            vertex: {
                module: device.createShaderModule({ code: blitSrc }),
                entryPoint: "vert",
                buffers: [
                    {
                        arrayStride: 8,
                        attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
                    },
                ],
            },
            fragment: {
                module: device.createShaderModule({ code: blitSrc }),
                entryPoint: "frag",
                targets: [{ format }],
            },
            primitive: { topology: "triangle-strip" },

            depthStencil: {
                depthWriteEnabled: false,
                depthCompare: "less",
                format: "depth24plus",
            },
        });

        // Fullscreen Quad Buffer
        this.fullscreenBuffer = device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.fullscreenBuffer, 0, new Float32Array([
            -1, -1, -1, 1, 1, -1, 1, 1,
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
        renderPassEncoder.setPipeline(this.blitPipeline);
        renderPassEncoder.setBindGroup(0, this.blitBindGroup);
        renderPassEncoder.setVertexBuffer(0, this.fullscreenBuffer);
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
                    resource: { buffer: this.volumetricBufferManager.densityGridBuffer },
                },
                {
                    binding: 1,
                    resource: this.volumetricBufferManager.outputTextureView,
                },
            ],
        });

        // 2. Blit Bind Group (reads from outputTexture)
        const blitBindGroupLayout = this.blitPipeline.getBindGroupLayout(0);
        this.blitBindGroup = device.createBindGroup({
            label: "volumetric blit bind group",
            layout: blitBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.volumetricBufferManager.outputTextureView,
                },
            ],
        });
    }
}
