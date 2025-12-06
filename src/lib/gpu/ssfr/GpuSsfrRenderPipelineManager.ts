import type { GpuUniformsBufferManager } from "$lib/gpu/uniforms/GpuUniformsBufferManager";
import ssfrDepthImpostorVertSrc from "./ssfrDepthImpostor.vert.wgsl?raw";
import ssfrDepthImpostorFragSrc from "./ssfrDepthImpostor.frag.wgsl?raw";
import ssfrNarrowRangeFilterSrc from "./ssfrNarrowRangeFilter.cs.wgsl?raw";
import ssfrNormalReconstructSrc from "./ssfrNormalReconstruct.cs.wgsl?raw";
import ssfrShadingSrc from "./ssfrShading.cs.wgsl?raw";
import ssfrCompositeVertSrc from "./ssfrComposite.vert.wgsl?raw";
import ssfrCompositeFragSrc from "./ssfrComposite.frag.wgsl?raw";
import ssfrThicknessVertSrc from "./ssfrThickness.vert.wgsl?raw";
import ssfrThicknessFragSrc from "./ssfrThickness.frag.wgsl?raw";
import ssfrSubsurfaceBlurSrc from "./ssfrSubsurfaceBlur.wgsl?raw";
import ssfrSubsurfaceCombineSrc from "./ssfrSubsurfaceCombine.wgsl?raw";
import type { GpuMpmBufferManager } from "../mpm/GpuMpmBufferManager";
import type { GpuRenderMethod } from "$lib/gpu/GpuRenderMethod";
import { attachPrelude } from "$lib/gpu/shaderPrelude";

export class GpuSsfrRenderPipelineManager implements GpuRenderMethod {
    readonly renderPipeline: GPURenderPipeline;

    readonly uniformsManager: GpuUniformsBufferManager;
    readonly mpmManager: GpuMpmBufferManager;
    
    private smoothedDepthTexture: GPUTexture | null = null;
    private smoothedDepthTextureView: GPUTextureView | null = null;
    private normalTexture: GPUTexture | null = null;
    private normalTextureView: GPUTextureView | null = null;
    private shadedOutputTexture: GPUTexture | null = null;
    private shadedOutputTextureView: GPUTextureView | null = null;
    private maskTexture: GPUTexture | null = null;
    private maskTextureView: GPUTextureView | null = null;

    private thicknessTexture: GPUTexture | null = null;
    private thicknessTextureView: GPUTextureView | null = null;
    private diffuseTexture: GPUTexture | null = null;
    private diffuseTextureView: GPUTextureView | null = null;
    private specularAmbientTexture: GPUTexture | null = null;
    private specularAmbientTextureView: GPUTextureView | null = null;
    private subsurfaceBlurTempTexture: GPUTexture | null = null;
    private subsurfaceBlurTempTextureView: GPUTextureView | null = null;

    private readonly bindGroup: GPUBindGroup;
    
    private readonly nrfComputePipeline: GPUComputePipeline;
    private nrfBindGroup: GPUBindGroup | null = null;
    private readonly nrfBindGroupLayout: GPUBindGroupLayout;
    
    private readonly normalReconstructPipeline: GPUComputePipeline;
    private normalReconstructBindGroup: GPUBindGroup | null = null;
    private readonly normalReconstructBindGroupLayout: GPUBindGroupLayout;
    
    private readonly shadingPipeline: GPUComputePipeline;
    private shadingBindGroup: GPUBindGroup | null = null;
    private readonly shadingBindGroupLayout: GPUBindGroupLayout;
    
    private readonly compositePipeline: GPURenderPipeline;
    private compositeBindGroup: GPUBindGroup | null = null;
    private readonly compositeBindGroupLayout: GPUBindGroupLayout;

    private readonly thicknessPipeline: GPURenderPipeline;
    private readonly thicknessBindGroup: GPUBindGroup;

    private readonly subsurfaceBlurHorizontalPipeline: GPUComputePipeline;
    private readonly subsurfaceBlurVerticalPipeline: GPUComputePipeline;
    private subsurfaceBlurHorizontalBindGroup: GPUBindGroup | null = null;
    private subsurfaceBlurVerticalBindGroup: GPUBindGroup | null = null;
    private readonly subsurfaceBlurBindGroupLayout: GPUBindGroupLayout;

    private readonly subsurfaceCombinePipeline: GPUComputePipeline;
    private subsurfaceCombineBindGroup: GPUBindGroup | null = null;
    private readonly subsurfaceCombineBindGroupLayout: GPUBindGroupLayout;
    
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
                        format: "rg32float",
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
                        format: "rg32float",
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
                {
                    binding: 4,
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

        // ================== Thickness Render Pipeline ==================
        const thicknessBindGroupLayout = device.createBindGroupLayout({
            label: "ssfr thickness bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage" },
                },
            ],
        });

        const thicknessVertModule = device.createShaderModule({
            label: "ssfr thickness vertex module",
            code: attachPrelude(ssfrThicknessVertSrc),
        });
        const thicknessFragModule = device.createShaderModule({
            label: "ssfr thickness fragment module",
            code: attachPrelude(ssfrThicknessFragSrc),
        });

        this.thicknessPipeline = device.createRenderPipeline({
            label: "ssfr thickness render pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [thicknessBindGroupLayout],
            }),
            vertex: {
                module: thicknessVertModule,
                entryPoint: "vert",
            },
            fragment: {
                module: thicknessFragModule,
                entryPoint: "frag",
                targets: [{
                    format: "rgba8unorm",
                    blend: {
                        color: { srcFactor: "one", dstFactor: "one", operation: "add" },
                        alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
                    },
                }],
            },
            primitive: {
                topology: "triangle-list",
            },
            // No depth testing for thickness accumulation
        });

        this.thicknessBindGroup = device.createBindGroup({
            label: "ssfr thickness bind group",
            layout: thicknessBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: uniformsManager.buffer } },
                { binding: 1, resource: { buffer: mpmManager.particleDataBuffer } },
            ],
        });

        // ================== SSS Blur Pipelines ==================
        const sssBlurBindGroupLayout = device.createBindGroupLayout({
            label: "ssfr sss blur bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { access: "write-only", format: "rgba8unorm", viewDimension: "2d" },
                },
            ],
        });
        this.subsurfaceBlurBindGroupLayout = sssBlurBindGroupLayout;

        const sssBlurModule = device.createShaderModule({
            label: "ssfr sss blur module",
            code: attachPrelude(ssfrSubsurfaceBlurSrc),
        });

        this.subsurfaceBlurHorizontalPipeline = device.createComputePipeline({
            label: "ssfr sss blur horizontal pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [sssBlurBindGroupLayout] }),
            compute: { module: sssBlurModule, entryPoint: "mainHorizontal" },
        });

        this.subsurfaceBlurVerticalPipeline = device.createComputePipeline({
            label: "ssfr sss blur vertical pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [sssBlurBindGroupLayout] }),
            compute: { module: sssBlurModule, entryPoint: "mainVertical" },
        });

        // ================== SSS Combine Pipeline ==================
        const sssCombineBindGroupLayout = device.createBindGroupLayout({
            label: "ssfr sss combine bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { access: "write-only", format: "rgba8unorm", viewDimension: "2d" },
                },
            ],
        });
        this.subsurfaceCombineBindGroupLayout = sssCombineBindGroupLayout;

        const sssCombineModule = device.createShaderModule({
            label: "ssfr sss combine module",
            code: attachPrelude(ssfrSubsurfaceCombineSrc),
        });

        this.subsurfaceCombinePipeline = device.createComputePipeline({
            label: "ssfr sss combine pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [sssCombineBindGroupLayout] }),
            compute: { module: sssCombineModule, entryPoint: "main" },
        });
    }

    resize(device: GPUDevice, width: number, height: number, depthTextureView: GPUTextureView) {
        this.destroyTextures();


        this.smoothedDepthTexture = device.createTexture({
            size: [width, height],
            format: "rg32float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.smoothedDepthTextureView = this.smoothedDepthTexture.createView();

        this.normalTexture = device.createTexture({
            size: [width, height],
            format: "rgba16float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.normalTextureView = this.normalTexture.createView();

        this.shadedOutputTexture = device.createTexture({
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.shadedOutputTextureView = this.shadedOutputTexture.createView();

        this.maskTexture = device.createTexture({
            size: [width, height],
            format: "rg32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.maskTextureView = this.maskTexture.createView();

        // SSS Textures (1x1 placeholders)
        this.thicknessTexture = device.createTexture({
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.thicknessTextureView = this.thicknessTexture.createView();

        this.diffuseTexture = device.createTexture({
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.diffuseTextureView = this.diffuseTexture.createView();

        this.specularAmbientTexture = device.createTexture({
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.specularAmbientTextureView = this.specularAmbientTexture.createView();

        this.subsurfaceBlurTempTexture = device.createTexture({
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.subsurfaceBlurTempTextureView = this.subsurfaceBlurTempTexture.createView();

        // NRF bind group
        this.nrfBindGroup = device.createBindGroup({
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
        this.normalReconstructBindGroup = device.createBindGroup({
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

        this.shadingBindGroup = device.createBindGroup({
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
                    resource: this.diffuseTextureView,
                },
                {
                    binding: 4,
                    resource: this.specularAmbientTextureView,
                },
            ],
        });

        // Composite bind group
        this.compositeBindGroup = device.createBindGroup({
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

        this.thicknessTexture = device.createTexture({
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.thicknessTextureView = this.thicknessTexture.createView();

        this.diffuseTexture = this.device.createTexture({
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.diffuseTextureView = this.diffuseTexture.createView();

        this.specularAmbientTexture = this.device.createTexture({
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.specularAmbientTextureView = this.specularAmbientTexture.createView();

        this.subsurfaceBlurTempTexture = this.device.createTexture({
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.subsurfaceBlurTempTextureView = this.subsurfaceBlurTempTexture.createView();

        this.shadingBindGroup = this.device.createBindGroup({
            label: "ssfr shading bind group",
            layout: this.shadingBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformsManager.buffer } },
                { binding: 1, resource: this.smoothedDepthTextureView },
                { binding: 2, resource: this.normalTextureView },
                { binding: 3, resource: this.diffuseTextureView },
                { binding: 4, resource: this.specularAmbientTextureView },
            ],
        });

        this.subsurfaceBlurHorizontalBindGroup = this.device.createBindGroup({
            label: "ssfr sss blur horizontal bind group",
            layout: this.subsurfaceBlurBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformsManager.buffer } },
                { binding: 1, resource: this.diffuseTextureView },
                { binding: 2, resource: this.thicknessTextureView },
                { binding: 3, resource: this.smoothedDepthTextureView },
                { binding: 4, resource: this.subsurfaceBlurTempTextureView },
            ],
        });

        this.subsurfaceBlurVerticalBindGroup = this.device.createBindGroup({
            label: "ssfr sss blur vertical bind group",
            layout: this.subsurfaceBlurBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformsManager.buffer } },
                { binding: 1, resource: this.subsurfaceBlurTempTextureView },
                { binding: 2, resource: this.thicknessTextureView },
                { binding: 3, resource: this.smoothedDepthTextureView },
                { binding: 4, resource: this.diffuseTextureView },
            ],
        });

        this.subsurfaceCombineBindGroup = this.device.createBindGroup({
            label: "ssfr sss combine bind group",
            layout: this.subsurfaceCombineBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformsManager.buffer } },
                { binding: 1, resource: this.diffuseTextureView },
                { binding: 2, resource: this.specularAmbientTextureView },
                { binding: 3, resource: this.smoothedDepthTextureView },
                { binding: 4, resource: this.shadedOutputTextureView },
            ],
        });
    }

    nPrerenderPasses(): number {
        return 3;
    }

    addPrerenderPasses(commandEncoder: GPUCommandEncoder, depthTextureView: GPUTextureView) {
        if (!this.maskTexture || !this.maskTextureView) return;
        if (!this.nrfBindGroup || !this.normalReconstructBindGroup || !this.shadingBindGroup) return;
        if (!this.subsurfaceBlurHorizontalBindGroup || !this.subsurfaceBlurVerticalBindGroup || !this.subsurfaceCombineBindGroup) return;
        if (!this.thicknessTexture || !this.diffuseTexture || !this.specularAmbientTexture || !this.subsurfaceBlurTempTexture) return;
        if (!this.smoothedDepthTexture || !this.smoothedDepthTextureView) return;
        if (!this.thicknessTextureView) return;

        const impostorPassEncoder = commandEncoder.beginRenderPass({
            label: "ssfr impostor render pass",
            colorAttachments: [
                {
                    view: this.maskTextureView,
                    clearValue: { r: 1.0, g: 0, b: 0, a: 0 },
                    loadOp: "clear",
                    storeOp: "store",
                }
            ],
            depthStencilAttachment: {
                view: depthTextureView,
                depthLoadOp: "load",
                depthStoreOp: "store",
            }
        });
        impostorPassEncoder.setBindGroup(0, this.bindGroup);
        impostorPassEncoder.setPipeline(this.renderPipeline);
        impostorPassEncoder.draw(6, this.mpmManager.nParticles, 0, 0);
        impostorPassEncoder.end();

        // thickness pass accumulates particle thickness with additive blending
        const thicknessPassDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [{
                view: this.thicknessTextureView,
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: "clear",
                storeOp: "store",
            }],
        };

        const thicknessPass = commandEncoder.beginRenderPass(thicknessPassDescriptor);
        thicknessPass.setPipeline(this.thicknessPipeline);
        thicknessPass.setBindGroup(0, this.thicknessBindGroup);
        thicknessPass.draw(6, this.mpmManager.nParticles, 0, 0);
        thicknessPass.end();

        // nrf, normal reconstruction, shading, sss blur, sss combine
        const width = this.smoothedDepthTexture.width;
        const height = this.smoothedDepthTexture.height;
        const workgroupsX = Math.ceil(width / 8);
        const workgroupsY = Math.ceil(height / 8);

        const computePass = commandEncoder.beginComputePass({
            label: "ssfr nrf compute pass",
        });
        computePass.setPipeline(this.nrfComputePipeline);
        computePass.setBindGroup(0, this.nrfBindGroup);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);

        computePass.setPipeline(this.normalReconstructPipeline);
        computePass.setBindGroup(0, this.normalReconstructBindGroup);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);

        computePass.setPipeline(this.shadingPipeline);
        computePass.setBindGroup(0, this.shadingBindGroup);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);

        computePass.setPipeline(this.subsurfaceBlurHorizontalPipeline);
        computePass.setBindGroup(0, this.subsurfaceBlurHorizontalBindGroup);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);

        computePass.setPipeline(this.subsurfaceBlurVerticalPipeline);
        computePass.setBindGroup(0, this.subsurfaceBlurVerticalBindGroup);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);

        computePass.setPipeline(this.subsurfaceCombinePipeline);
        computePass.setBindGroup(0, this.subsurfaceCombineBindGroup);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);

        computePass.end();
    }

    addFinalDraw(renderPassEncoder: GPURenderPassEncoder) {
        this.addCompositePass(renderPassEncoder);
    }
    

    addCompositePass(renderPassEncoder: GPURenderPassEncoder) {
        if (!this.compositeBindGroup) return;
        
        renderPassEncoder.setPipeline(this.compositePipeline);
        renderPassEncoder.setBindGroup(0, this.compositeBindGroup);
        renderPassEncoder.draw(4, 1, 0, 0);
    }

    private destroyTextures() {
        this.smoothedDepthTexture?.destroy();
        this.normalTexture?.destroy();
        this.shadedOutputTexture?.destroy();
        this.maskTexture?.destroy();
        this.thicknessTexture?.destroy();
        this.diffuseTexture?.destroy();
        this.specularAmbientTexture?.destroy();
        this.subsurfaceBlurTempTexture?.destroy();
    }


    destroy() {
        this.destroyTextures();
    }
}
