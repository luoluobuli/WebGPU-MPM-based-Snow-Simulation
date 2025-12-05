import type { GpuUniformsBufferManager } from "$lib/gpu/uniforms/GpuUniformsBufferManager";
import ssfrDepthImpostorVertSrc from "./ssfrDepthImpostor.vert.wgsl?raw";
import ssfrDepthImpostorFragSrc from "./ssfrDepthImpostor.frag.wgsl?raw";
import ssfrNarrowRangeFilterSrc from "./ssfrNarrowRangeFilter.cs.wgsl?raw";
import ssfrNormalReconstructSrc from "./ssfrNormalReconstruct.cs.wgsl?raw";
import ssfrShadingSrc from "./ssfrShading.cs.wgsl?raw";
import ssfrCompositeVertSrc from "./ssfrComposite.vert.wgsl?raw";
import ssfrCompositeFragSrc from "./ssfrComposite.frag.wgsl?raw";
import type { GpuMpmBufferManager } from "../mpm/GpuMpmBufferManager";
import type { GpuRenderMethod } from "$lib/gpu/GpuRenderMethod";
import { attachPrelude } from "$lib/gpu/shaderPrelude";

export class GpuSsfrRenderPipelineManager implements GpuRenderMethod {
    readonly renderPipeline: GPURenderPipeline;

    readonly uniformsManager: GpuUniformsBufferManager;
    readonly mpmManager: GpuMpmBufferManager;
    
    // Textures
    smoothedDepthTexture: GPUTexture;
    smoothedDepthTextureView: GPUTextureView;
    private normalTexture: GPUTexture;
    private normalTextureView: GPUTextureView;
    private shadedOutputTexture: GPUTexture;
    private shadedOutputTextureView: GPUTextureView;
    maskTexture: GPUTexture;
    maskTextureView: GPUTextureView;

    private readonly bindGroup: GPUBindGroup;
    
    // NRF (Narrow Range Filter) pipeline
    private readonly nrfComputePipeline: GPUComputePipeline;
    private nrfBindGroup: GPUBindGroup | null = null;
    private readonly nrfBindGroupLayout: GPUBindGroupLayout;
    
    // Normal reconstruction pipeline
    private readonly normalReconstructPipeline: GPUComputePipeline;
    private normalReconstructBindGroup: GPUBindGroup | null = null;
    private readonly normalReconstructBindGroupLayout: GPUBindGroupLayout;
    
    // Shading pipeline
    private readonly shadingPipeline: GPUComputePipeline;
    private shadingBindGroup: GPUBindGroup | null = null;
    private readonly shadingBindGroupLayout: GPUBindGroupLayout;
    
    // Composite render pipeline
    private readonly compositePipeline: GPURenderPipeline;
    private compositeBindGroup: GPUBindGroup | null = null;
    private readonly compositeBindGroupLayout: GPUBindGroupLayout;
    private readonly compositeVertBuffer: GPUBuffer;
    
    private readonly device: GPUDevice;

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
        this.device = device;
        this.uniformsManager = uniformsManager;
        this.mpmManager = mpmManager;

        // ================== Depth Impostor Render Pipeline ==================
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
                        format: "r32float", // Mask texture format (using r32float for high precision depth)
                        writeMask: GPUColorWrite.ALL,
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

        this.bindGroup = bindGroup;

        // ================== NRF Compute Pipeline ==================
        const nrfBindGroupLayout = device.createBindGroupLayout({
            label: "ssfr nrf bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform",
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: {
                        sampleType: "depth",
                        viewDimension: "2d",
                        multisampled: false,
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        access: "write-only",
                        format: "r32float",
                        viewDimension: "2d",
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: {
                        sampleType: "unfilterable-float",
                        viewDimension: "2d",
                        multisampled: false,
                    }
                },
            ],
        });
        this.nrfBindGroupLayout = nrfBindGroupLayout;

        const nrfModule = device.createShaderModule({
            label: "ssfr nrf shader module",
            code: attachPrelude(ssfrNarrowRangeFilterSrc),
        });

        const nrfPipelineLayout = device.createPipelineLayout({
            label: "ssfr nrf pipeline layout",
            bindGroupLayouts: [nrfBindGroupLayout],
        });

        this.nrfComputePipeline = device.createComputePipeline({
            label: "ssfr nrf compute pipeline",
            layout: nrfPipelineLayout,
            compute: {
                module: nrfModule,
                entryPoint: "main",
            },
        });

        // ================== Normal Reconstruction Pipeline ==================
        const normalReconstructBindGroupLayout = device.createBindGroupLayout({
            label: "ssfr normal reconstruct bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: {
                        sampleType: "unfilterable-float",
                        viewDimension: "2d",
                    },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        access: "write-only",
                        format: "rgba16float",
                        viewDimension: "2d",
                    },
                },
            ],
        });
        this.normalReconstructBindGroupLayout = normalReconstructBindGroupLayout;

        const normalReconstructModule = device.createShaderModule({
            label: "ssfr normal reconstruct module",
            code: attachPrelude(ssfrNormalReconstructSrc),
        });

        this.normalReconstructPipeline = device.createComputePipeline({
            label: "ssfr normal reconstruct pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [normalReconstructBindGroupLayout],
            }),
            compute: {
                module: normalReconstructModule,
                entryPoint: "main",
            },
        });

        // ================== Shading Pipeline ==================
        const shadingBindGroupLayout = device.createBindGroupLayout({
            label: "ssfr shading bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: {
                        sampleType: "unfilterable-float",
                        viewDimension: "2d",
                    },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: {
                        sampleType: "unfilterable-float",
                        viewDimension: "2d",
                    },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        access: "write-only",
                        format: "rgba8unorm",
                        viewDimension: "2d",
                    },
                },
            ],
        });
        this.shadingBindGroupLayout = shadingBindGroupLayout;

        const shadingModule = device.createShaderModule({
            label: "ssfr shading module",
            code: attachPrelude(ssfrShadingSrc),
        });

        this.shadingPipeline = device.createComputePipeline({
            label: "ssfr shading pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [shadingBindGroupLayout],
            }),
            compute: {
                module: shadingModule,
                entryPoint: "main",
            },
        });

        // ================== Composite Render Pipeline ==================
        const compositeBindGroupLayout = device.createBindGroupLayout({
            label: "ssfr composite bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: {
                        sampleType: "float",
                        viewDimension: "2d",
                    },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: {
                        sampleType: "unfilterable-float",
                        viewDimension: "2d",
                    },
                },
            ],
        });
        this.compositeBindGroupLayout = compositeBindGroupLayout;

        const compositeVertModule = device.createShaderModule({
            label: "ssfr composite vertex module",
            code: ssfrCompositeVertSrc,
        });
        const compositeFragModule = device.createShaderModule({
            label: "ssfr composite fragment module",
            code: attachPrelude(ssfrCompositeFragSrc),
        });

        this.compositePipeline = device.createRenderPipeline({
            label: "ssfr composite render pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [compositeBindGroupLayout],
            }),
            vertex: {
                module: compositeVertModule,
                entryPoint: "vert",
            },
            fragment: {
                module: compositeFragModule,
                entryPoint: "frag",
                targets: [{ format }],
            },
            primitive: {
                topology: "triangle-strip",
            },
            depthStencil: {
                depthWriteEnabled: false,
                depthCompare: "less-equal",
                format: depthFormat,
            },
        });

        // Vertex buffer for fullscreen quad (not actually needed, using vertex_index)
        this.compositeVertBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.VERTEX,
        });

        // ================== Initialize Textures (1x1 placeholders) ==================
        this.smoothedDepthTexture = device.createTexture({
            size: [1, 1],
            format: "r32float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.smoothedDepthTextureView = this.smoothedDepthTexture.createView();

        this.normalTexture = device.createTexture({
            size: [1, 1],
            format: "rgba16float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.normalTextureView = this.normalTexture.createView();

        this.shadedOutputTexture = device.createTexture({
            size: [1, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.shadedOutputTextureView = this.shadedOutputTexture.createView();

        this.maskTexture = device.createTexture({
            size: [1, 1],
            format: "r32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.maskTextureView = this.maskTexture.createView();
    }

    resize(width: number, height: number, depthTextureView: GPUTextureView) {
        // Recreate smoothed depth texture
        this.smoothedDepthTexture = this.device.createTexture({
            size: [width, height],
            format: "r32float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.smoothedDepthTextureView = this.smoothedDepthTexture.createView();

        // Recreate normal texture
        this.normalTexture = this.device.createTexture({
            size: [width, height],
            format: "rgba16float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.normalTextureView = this.normalTexture.createView();

        // Recreate shaded output texture
        this.shadedOutputTexture = this.device.createTexture({
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.shadedOutputTextureView = this.shadedOutputTexture.createView();

        // Recreate mask texture (high precision depth)
        this.maskTexture = this.device.createTexture({
            size: [width, height],
            format: "r32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.maskTextureView = this.maskTexture.createView();

        // NRF bind group
        this.nrfBindGroup = this.device.createBindGroup({
            label: "ssfr nrf bind group",
            layout: this.nrfBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.uniformsManager.buffer },
                },
                {
                    binding: 1,
                    resource: depthTextureView,
                },
                {
                    binding: 2,
                    resource: this.smoothedDepthTextureView,
                },
                {
                    binding: 3,
                    resource: this.maskTextureView,
                },
            ],
        });

        // Normal reconstruction bind group
        this.normalReconstructBindGroup = this.device.createBindGroup({
            label: "ssfr normal reconstruct bind group",
            layout: this.normalReconstructBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.uniformsManager.buffer },
                },
                {
                    binding: 1,
                    resource: this.smoothedDepthTextureView,
                },
                {
                    binding: 2,
                    resource: this.normalTextureView,
                },
            ],
        });

        // Shading bind group
        this.shadingBindGroup = this.device.createBindGroup({
            label: "ssfr shading bind group",
            layout: this.shadingBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.uniformsManager.buffer },
                },
                {
                    binding: 1,
                    resource: this.smoothedDepthTextureView,
                },
                {
                    binding: 2,
                    resource: this.normalTextureView,
                },
                {
                    binding: 3,
                    resource: this.shadedOutputTextureView,
                },
            ],
        });

        // Composite bind group
        this.compositeBindGroup = this.device.createBindGroup({
            label: "ssfr composite bind group",
            layout: this.compositeBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.uniformsManager.buffer },
                },
                {
                    binding: 1,
                    resource: this.shadedOutputTextureView,
                },
                {
                    binding: 2,
                    resource: this.smoothedDepthTextureView,
                },
            ],
        });
    }

    addComputePasses(commandEncoder: GPUCommandEncoder) {
        // Skip if bind groups not yet initialized (before first resize)
        if (!this.nrfBindGroup || !this.normalReconstructBindGroup || !this.shadingBindGroup) return;

        const width = this.smoothedDepthTexture.width;
        const height = this.smoothedDepthTexture.height;
        const workgroupsX = Math.ceil(width / 8);
        const workgroupsY = Math.ceil(height / 8);

        // Pass 1: Narrow Range Filter (depth smoothing)
        const nrfPass = commandEncoder.beginComputePass({
            label: "ssfr nrf compute pass",
        });
        nrfPass.setPipeline(this.nrfComputePipeline);
        nrfPass.setBindGroup(0, this.nrfBindGroup);
        nrfPass.dispatchWorkgroups(workgroupsX, workgroupsY);
        nrfPass.end();

        // Pass 2: Normal Reconstruction
        const normalPass = commandEncoder.beginComputePass({
            label: "ssfr normal reconstruct compute pass",
        });
        normalPass.setPipeline(this.normalReconstructPipeline);
        normalPass.setBindGroup(0, this.normalReconstructBindGroup);
        normalPass.dispatchWorkgroups(workgroupsX, workgroupsY);
        normalPass.end();

        // Pass 3: Shading with noise injection
        const shadingPass = commandEncoder.beginComputePass({
            label: "ssfr shading compute pass",
        });
        shadingPass.setPipeline(this.shadingPipeline);
        shadingPass.setBindGroup(0, this.shadingBindGroup);
        shadingPass.dispatchWorkgroups(workgroupsX, workgroupsY);
        shadingPass.end();
    }

    addImpostorPass(renderPassEncoder: GPURenderPassEncoder) {
        renderPassEncoder.setBindGroup(0, this.bindGroup);
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.draw(6, this.mpmManager.nParticles, 0, 0);
    }

    addDraw(renderPassEncoder: GPURenderPassEncoder) {
        // First draw the depth impostors
        this.addImpostorPass(renderPassEncoder);
    }

    addCompositePass(renderPassEncoder: GPURenderPassEncoder) {
        if (!this.compositeBindGroup) return;
        
        renderPassEncoder.setPipeline(this.compositePipeline);
        renderPassEncoder.setBindGroup(0, this.compositeBindGroup);
        renderPassEncoder.draw(4, 1, 0, 0);
    }
}
