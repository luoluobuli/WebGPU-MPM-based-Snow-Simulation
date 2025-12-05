import type { GpuUniformsBufferManager } from "$lib/gpu/uniforms/GpuUniformsBufferManager";
import type { GpuMpmBufferManager } from "../mpm/GpuMpmBufferManager";
import type { GpuRenderMethod } from "$lib/gpu/GpuRenderMethod";
import { attachPrelude } from "$lib/gpu/shaderPrelude";
import { GpuMarchingCubesBufferManager } from "./GpuMarchingCubesBufferManager";

import preludeSrc from "./prelude.wgsl?raw";
import triTableSrc from "./triTable.wgsl?raw";
import mcDensitySrc from "./mcDensity.cs.wgsl?raw";
import mcVertexDensitySrc from "./mcVertexDensity.cs.wgsl?raw";
import mcGenerateSrc from "./mcGenerate.cs.wgsl?raw";
import mcRenderVertSrc from "./mcRender.vert.wgsl?raw";
import mcRenderFragSrc from "./mcRender.frag.wgsl?raw";
import mcShadingSrc from "./mcShading.cs.wgsl?raw";
import mcCompositeVertSrc from "./mcComposite.vert.wgsl?raw";
import mcCompositeFragSrc from "./mcComposite.frag.wgsl?raw";

export class GpuMarchingCubesRenderPipelineManager implements GpuRenderMethod {
    private readonly device: GPUDevice;
    private readonly uniformsManager: GpuUniformsBufferManager;
    private readonly mpmManager: GpuMpmBufferManager;
    private readonly bufferManager: GpuMarchingCubesBufferManager;
    
    // Compute pipelines
    private readonly densityPipeline: GPUComputePipeline;
    private readonly vertexDensityPipeline: GPUComputePipeline;
    private readonly generatePipeline: GPUComputePipeline;
    private readonly shadingPipeline: GPUComputePipeline;
    
    // Render pipelines
    private readonly meshRenderPipeline: GPURenderPipeline;
    private readonly compositePipeline: GPURenderPipeline;
    
    // Bind groups
    private densityBindGroup: GPUBindGroup;
    private vertexDensityBindGroup: GPUBindGroup;
    private generateBindGroup: GPUBindGroup;
    private shadingBindGroup: GPUBindGroup;
    private compositeBindGroup: GPUBindGroup;
    
    // Textures for G-buffer
    private normalTexture: GPUTexture;
    private normalTextureView: GPUTextureView;
    private albedoTexture: GPUTexture;
    private albedoTextureView: GPUTextureView;
    private shadedTexture: GPUTexture;
    private shadedTextureView: GPUTextureView;
    private mcDepthTexture: GPUTexture;
    private mcDepthTextureView: GPUTextureView;
    
    private screenWidth: number = 1;
    private screenHeight: number = 1;
    
    // Bind group layouts for recreation on resize
    private readonly shadingBindGroupLayout: GPUBindGroupLayout;
    private readonly compositeBindGroupLayout: GPUBindGroupLayout;
    
    // MC parameters buffer
    private readonly mcParamsBuffer: GPUBuffer;
    
    constructor({
        device,
        format,
        depthFormat,
        uniformsManager,
        mpmManager,
        gridResolutionX,
        gridResolutionY,
        gridResolutionZ,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        depthFormat: GPUTextureFormat,
        uniformsManager: GpuUniformsBufferManager,
        mpmManager: GpuMpmBufferManager,
        gridResolutionX: number,
        gridResolutionY: number,
        gridResolutionZ: number,
    }) {
        this.device = device;
        this.uniformsManager = uniformsManager;
        this.mpmManager = mpmManager;
        
        this.bufferManager = new GpuMarchingCubesBufferManager({
            device,
            gridResolutionX,
            gridResolutionY,
            gridResolutionZ,
        });
        
        // Create MC params uniform buffer
        const mcParamsBuffer = device.createBuffer({
            label: "MC params buffer",
            size: 16, // vec3u + u32 = 16 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.mcParamsBuffer = mcParamsBuffer;
        
        // Write MC params
        const [mcX, mcY, mcZ] = this.bufferManager.mcGridDims;
        device.queue.writeBuffer(mcParamsBuffer, 0, new Uint32Array([mcX, mcY, mcZ, this.bufferManager.downsampleFactor]));
        
        const mcPrelude = `${preludeSrc}\n${triTableSrc}`;
        
        // === Density computation pipeline ===
        const densityBindGroupLayout = device.createBindGroupLayout({
            label: "MC density bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE,
                  buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE,
                  buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE,
                  buffer: { type: "uniform" } },
            ],
        });
        
        this.densityPipeline = device.createComputePipeline({
            label: "MC density pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [uniformsManager.bindGroupLayout, densityBindGroupLayout],
            }),
            compute: {
                module: device.createShaderModule({
                    code: attachPrelude(mcDensitySrc),
                }),
                entryPoint: "calculateDensity",
            },
        });
        
        this.densityBindGroup = device.createBindGroup({
            label: "MC density bind group",
            layout: densityBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: mpmManager.particleDataBuffer } },
                { binding: 1, resource: { buffer: this.bufferManager.densityGridBuffer } },
                { binding: 2, resource: { buffer: mcParamsBuffer } },
            ],
        });
        
        // === Vertex density pipeline ===
        const vertexDensityBindGroupLayout = device.createBindGroupLayout({
            label: "MC vertex density bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE,
                  buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE,
                  buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE,
                  buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE,
                  buffer: { type: "uniform" } },
            ],
        });
        
        this.vertexDensityPipeline = device.createComputePipeline({
            label: "MC vertex density pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [uniformsManager.bindGroupLayout, vertexDensityBindGroupLayout],
            }),
            compute: {
                module: device.createShaderModule({
                    code: attachPrelude(mcVertexDensitySrc),
                }),
                entryPoint: "computeVertexDensity",
            },
        });
        
        this.vertexDensityBindGroup = device.createBindGroup({
            label: "MC vertex density bind group",
            layout: vertexDensityBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.densityGridBuffer } },
                { binding: 1, resource: { buffer: this.bufferManager.vertexDensityBuffer } },
                { binding: 2, resource: { buffer: this.bufferManager.vertexGradientBuffer } },
                { binding: 3, resource: { buffer: mcParamsBuffer } },
            ],
        });
        
        // === Generate mesh pipeline ===
        const generateBindGroupLayout = device.createBindGroupLayout({
            label: "MC generate bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE,
                  buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE,
                  buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE,
                  buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE,
                  buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE,
                  buffer: { type: "uniform" } },
            ],
        });
        
        this.generatePipeline = device.createComputePipeline({
            label: "MC generate pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [uniformsManager.bindGroupLayout, generateBindGroupLayout],
            }),
            compute: {
                module: device.createShaderModule({
                    code: attachPrelude(`${mcPrelude}\n${mcGenerateSrc}`),
                }),
                entryPoint: "generateMesh",
            },
        });
        
        this.generateBindGroup = device.createBindGroup({
            label: "MC generate bind group",
            layout: generateBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.vertexDensityBuffer } },
                { binding: 1, resource: { buffer: this.bufferManager.vertexGradientBuffer } },
                { binding: 2, resource: { buffer: this.bufferManager.vertexBuffer } },
                { binding: 3, resource: { buffer: this.bufferManager.indirectDrawBuffer } },
                { binding: 4, resource: { buffer: mcParamsBuffer } },
            ],
        });
        
        // === Mesh render pipeline ===
        this.meshRenderPipeline = device.createRenderPipeline({
            label: "MC mesh render pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [uniformsManager.bindGroupLayout],
            }),
            vertex: {
                module: device.createShaderModule({
                    code: attachPrelude(mcRenderVertSrc),
                }),
                entryPoint: "vert",
                buffers: [{
                    arrayStride: 32, // 2 * vec3f aligned to 16 bytes each => 32 bytes total
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: "float32x3" as GPUVertexFormat },
                        { shaderLocation: 1, offset: 16, format: "float32x3" as GPUVertexFormat },
                    ],
                }],
            },
            fragment: {
                module: device.createShaderModule({
                    code: attachPrelude(mcRenderFragSrc),
                }),
                entryPoint: "frag",
                targets: [
                    { format }, // albedo
                    { format: "rgba16float" }, // normal
                ],
            },
            primitive: {
                topology: "triangle-list",
                cullMode: "none", // Need both faces for now
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: depthFormat,
            },
        });
        
        // Create initial textures (will be recreated on resize)
        this.normalTexture = device.createTexture({
            label: "MC normal texture",
            size: [1, 1],
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.normalTextureView = this.normalTexture.createView();
        
        this.shadedTexture = device.createTexture({
            label: "MC shaded texture",
            size: [1, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.shadedTextureView = this.shadedTexture.createView();
        
        this.mcDepthTexture = device.createTexture({
            label: "MC depth texture",
            size: [1, 1],
            format: depthFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.mcDepthTextureView = this.mcDepthTexture.createView();
        
        // === Shading pipeline ===
        this.shadingBindGroupLayout = device.createBindGroupLayout({
            label: "MC shading bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE,
                  buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE,
                  texture: { sampleType: "depth" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE,
                  texture: { sampleType: "float" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE,
                  storageTexture: { access: "write-only", format: "rgba8unorm" } },
            ],
        });
        
        this.shadingPipeline = device.createComputePipeline({
            label: "MC shading pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [this.shadingBindGroupLayout],
            }),
            compute: {
                module: device.createShaderModule({
                    code: attachPrelude(mcShadingSrc),
                }),
                entryPoint: "main",
            },
        });
        
        this.shadingBindGroup = this.createShadingBindGroup();
        
        // === Composite pipeline ===
        this.compositeBindGroupLayout = device.createBindGroupLayout({
            label: "MC composite bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT,
                  texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT,
                  texture: { sampleType: "depth" } },
            ],
        });
        
        this.compositePipeline = device.createRenderPipeline({
            label: "MC composite pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [this.compositeBindGroupLayout],
            }),
            vertex: {
                module: device.createShaderModule({
                    code: mcCompositeVertSrc,
                }),
                entryPoint: "vert",
            },
            fragment: {
                module: device.createShaderModule({
                    code: mcCompositeFragSrc,
                }),
                entryPoint: "frag",
                targets: [{ format }],
            },
            primitive: { topology: "triangle-strip" },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "always",
                format: depthFormat,
            },
        });
        
        this.compositeBindGroup = this.createCompositeBindGroup();
    }
    
    private createShadingBindGroup(): GPUBindGroup {
        return this.device.createBindGroup({
            label: "MC shading bind group",
            layout: this.shadingBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformsManager.buffer } },
                { binding: 1, resource: this.mcDepthTextureView },
                { binding: 2, resource: this.normalTextureView },
                { binding: 3, resource: this.shadedTextureView },
            ],
        });
    }
    
    private createCompositeBindGroup(): GPUBindGroup {
        return this.device.createBindGroup({
            label: "MC composite bind group",
            layout: this.compositeBindGroupLayout,
            entries: [
                { binding: 0, resource: this.shadedTextureView },
                { binding: 1, resource: this.mcDepthTextureView },
            ],
        });
    }
    
    resize(width: number, height: number) {
        this.screenWidth = width;
        this.screenHeight = height;
        
        this.normalTexture.destroy();
        this.normalTexture = this.device.createTexture({
            label: "MC normal texture",
            size: [width, height],
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });
        this.normalTextureView = this.normalTexture.createView();
        
        if (this.albedoTexture) this.albedoTexture.destroy();
        this.albedoTexture = this.device.createTexture({
            label: "MC albedo texture",
            size: [width, height],
            format: "bgra8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.albedoTextureView = this.albedoTexture.createView();
        
        this.shadedTexture.destroy();
        this.shadedTexture = this.device.createTexture({
            label: "MC shaded texture",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.shadedTextureView = this.shadedTexture.createView();
        
        this.mcDepthTexture.destroy();
        this.mcDepthTexture = this.device.createTexture({
            label: "MC depth texture",
            size: [width, height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.mcDepthTextureView = this.mcDepthTexture.createView();
        
        this.shadingBindGroup = this.createShadingBindGroup();
        this.compositeBindGroup = this.createCompositeBindGroup();
    }
    
    addComputePasses(commandEncoder: GPUCommandEncoder) {
        // Clear density buffer (but NOT the indirect draw buffer - that would race with queue.writeBuffer)
        commandEncoder.clearBuffer(this.bufferManager.densityGridBuffer);
        
        // Write initial indirect draw args: vertexCount=0 (will be filled by shader), rest fixed
        // This must be done via queue.writeBuffer BEFORE the command encoder is submitted
        this.device.queue.writeBuffer(
            this.bufferManager.indirectDrawBuffer,
            0,
            new Uint32Array([0, 1, 0, 0]) // vertexCount=0, instanceCount=1, firstVertex=0, firstInstance=0
        );
        
        const computePass = commandEncoder.beginComputePass({
            label: "MC compute pass",
        });
        
        // 1. Calculate density grid
        computePass.setPipeline(this.densityPipeline);
        computePass.setBindGroup(0, this.uniformsManager.bindGroup);
        computePass.setBindGroup(1, this.densityBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.mpmManager.nParticles / 256));
        
        // 2. Calculate vertex densities and gradients
        computePass.setPipeline(this.vertexDensityPipeline);
        computePass.setBindGroup(0, this.uniformsManager.bindGroup);
        computePass.setBindGroup(1, this.vertexDensityBindGroup);
        const [gx, gy, gz] = this.bufferManager.gridDims;
        computePass.dispatchWorkgroups(
            Math.ceil((gx + 1) / 8),
            Math.ceil((gy + 1) / 8),
            Math.ceil((gz + 1) / 4)
        );
        
        // 3. Generate mesh
        computePass.setPipeline(this.generatePipeline);
        computePass.setBindGroup(0, this.uniformsManager.bindGroup);
        computePass.setBindGroup(1, this.generateBindGroup);
        computePass.dispatchWorkgroups(
            Math.ceil(gx / 8),
            Math.ceil(gy / 8),
            Math.ceil(gz / 4)
        );
        
        computePass.end();
    }
    
    addMeshRenderPass(commandEncoder: GPUCommandEncoder, depthTextureView: GPUTextureView) {
        const renderPass = commandEncoder.beginRenderPass({
            label: "MC mesh render pass",
            colorAttachments: [
                {
                    view: this.albedoTextureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: "clear",
                    storeOp: "store",
                },
                {
                    view: this.normalTextureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: "clear",
                    storeOp: "store",
                },
            ],
            depthStencilAttachment: {
                view: this.mcDepthTextureView,
                depthLoadOp: "clear",
                depthStoreOp: "store",
                depthClearValue: 1.0,
            },
        });
        
        renderPass.setPipeline(this.meshRenderPipeline);
        renderPass.setBindGroup(0, this.uniformsManager.bindGroup);
        renderPass.setVertexBuffer(0, this.bufferManager.vertexBuffer);
        // DEBUG: Use indirect draw with forced vertex count
        renderPass.drawIndirect(this.bufferManager.indirectDrawBuffer, 0);
        renderPass.end();
    }
    
    addShadingPass(commandEncoder: GPUCommandEncoder) {
        const computePass = commandEncoder.beginComputePass({
            label: "MC shading pass",
        });
        
        computePass.setPipeline(this.shadingPipeline);
        computePass.setBindGroup(0, this.shadingBindGroup);
        computePass.dispatchWorkgroups(
            Math.ceil(this.screenWidth / 8),
            Math.ceil(this.screenHeight / 8)
        );
        
        computePass.end();
    }
    
    addCompositePass(renderPassEncoder: GPURenderPassEncoder) {
        renderPassEncoder.setPipeline(this.compositePipeline);
        renderPassEncoder.setBindGroup(0, this.compositeBindGroup);
        renderPassEncoder.draw(4);
    }
    
    // Required by GpuRenderMethod interface
    addDraw(renderPassEncoder: GPURenderPassEncoder): void {
        // Marching cubes uses its own render passes, this is a no-op
    }
}
