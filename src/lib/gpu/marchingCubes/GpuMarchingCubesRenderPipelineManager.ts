import type { GpuUniformsBufferManager } from "$lib/gpu/uniforms/GpuUniformsBufferManager";
import type { GpuMpmBufferManager } from "../mpm/GpuMpmBufferManager";
import type { GpuRenderMethod } from "$lib/gpu/GpuRenderMethod";
import { attachPrelude } from "$lib/gpu/shaderPrelude";
import { GpuMarchingCubesBufferManager } from "./GpuMarchingCubesBufferManager";

import preludeSrc from "./prelude.wgsl?raw";
import triTableSrc from "./triTable.wgsl?raw";
import mcDensitySrc from "./mcDensity.cs.wgsl?raw";
import mcListBlocksSrc from "./mcListBlocks.cs.wgsl?raw";
import mcVertexDensitySrc from "./mcVertexDensity.cs.wgsl?raw";
import mcGenerateSrc from "./mcGenerate.cs.wgsl?raw";
import mcRenderVertSrc from "./mcRender.vert.wgsl?raw";
import mcRenderFragSrc from "./mcRender.frag.wgsl?raw";
import mcShadingSrc from "./mcShading.cs.wgsl?raw";
import mcCompositeVertSrc from "./mcComposite.vert.wgsl?raw";
import mcCompositeFragSrc from "./mcComposite.frag.wgsl?raw";
import mcResetSrc from "./mcReset.cs.wgsl?raw";
import mcClampArgsSrc from "./mcClampArgs.cs.wgsl?raw";

export class GpuMarchingCubesRenderPipelineManager implements GpuRenderMethod {
    private readonly device: GPUDevice;
    private readonly uniformsManager: GpuUniformsBufferManager;
    private readonly mpmManager: GpuMpmBufferManager;
    private readonly bufferManager: GpuMarchingCubesBufferManager;
    
    // Compute pipelines
    private readonly densityPipeline: GPUComputePipeline;
    private readonly listBlocksPipeline: GPUComputePipeline;
    private readonly vertexDensityPipeline: GPUComputePipeline;
    private readonly generatePipeline: GPUComputePipeline;
    private readonly shadingPipeline: GPUComputePipeline;
    private readonly resetPipeline: GPUComputePipeline;
    private readonly clampArgsPipeline: GPUComputePipeline;

    // Render pipelines
    private readonly meshRenderPipeline: GPURenderPipeline;
    private readonly compositePipeline: GPURenderPipeline;

    // Bind groups
    private densityBindGroup: GPUBindGroup;
    private listBlocksBindGroup: GPUBindGroup;
    private vertexDensityBindGroup: GPUBindGroup;
    private generateBindGroup: GPUBindGroup;
    private shadingBindGroup: GPUBindGroup;
    private compositeBindGroup: GPUBindGroup;
    private resetBindGroup: GPUBindGroup;
    private clampArgsBindGroup: GPUBindGroup;
    
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
    private readonly maxVertsBuffer: GPUBuffer;
    
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

        // Max verts uniform for clamping
        // 3.5M * 3 = 10,500,000 vertices
        const MAX_TOTAL_VERTICES = 10500000;
        this.maxVertsBuffer = device.createBuffer({
            label: "MC max verts buffer",
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.maxVertsBuffer, 0, new Uint32Array([MAX_TOTAL_VERTICES]));
        
        // Write MC params
        const [mcX, mcY, mcZ] = this.bufferManager.marchingCubesGridDims;
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
        
        // === List blocks pipeline ===
        const listBlocksBindGroupLayout = device.createBindGroupLayout({
            label: "MC list blocks bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // densityGrid
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // activeBlocks
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // indirectDispatch
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },           // mcParams
            ],
        });

        this.listBlocksPipeline = device.createComputePipeline({
            label: "MC list blocks pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [listBlocksBindGroupLayout],
            }),
            compute: {
                module: device.createShaderModule({
                    code: mcListBlocksSrc,
                }),
                entryPoint: "listBlocks",
            },
        });

        this.listBlocksBindGroup = device.createBindGroup({
            label: "MC list blocks bind group",
            layout: listBlocksBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.densityGridBuffer } },
                { binding: 1, resource: { buffer: this.bufferManager.activeBlocksBuffer } },
                { binding: 2, resource: { buffer: this.bufferManager.blockIndirectDispatchBuffer } },
                { binding: 3, resource: { buffer: mcParamsBuffer } },
            ],
        });

        // === Vertex density pipeline ===
        const vertexDensityBindGroupLayout = device.createBindGroupLayout({
            label: "MC vertex density bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // densityGrid
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // vertexDensity
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // vertexGradient
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },           // mcParams
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // activeBlocks
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
                { binding: 4, resource: { buffer: this.bufferManager.activeBlocksBuffer } },
            ],
        });
        
        // === Generate mesh pipeline ===
        const generateBindGroupLayout = device.createBindGroupLayout({
            label: "MC generate bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // vertexDensity
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // vertexGradient
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // outputVertices
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // indirectDraw
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },           // mcParams
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // activeBlocks
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
                { binding: 5, resource: { buffer: this.bufferManager.activeBlocksBuffer } },
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
                    arrayStride: 24, // 2 * vec3f packed = 24 bytes
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: "float32x3" as GPUVertexFormat },
                        { shaderLocation: 1, offset: 12, format: "float32x3" as GPUVertexFormat },
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
        
        this.albedoTexture = device.createTexture({
            label: "MC albedo texture",
            size: [1, 1],
            format: format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.albedoTextureView = this.albedoTexture.createView();

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

        // === Reset pipeline ===
        const resetBindGroupLayout = device.createBindGroupLayout({
            label: "MC reset bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // indirectDispatch
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // indirectDraw
            ],
        });

        this.resetPipeline = device.createComputePipeline({
            label: "MC reset pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [resetBindGroupLayout],
            }),
            compute: {
                module: device.createShaderModule({
                    code: mcResetSrc,
                }),
                entryPoint: "main",
            },
        });

        this.resetBindGroup = device.createBindGroup({
            label: "MC reset bind group",
            layout: resetBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.blockIndirectDispatchBuffer } },
                { binding: 1, resource: { buffer: this.bufferManager.indirectDrawBuffer } },
            ],
        });

        // === Clamp Args Pipeline ===
        const clampBindGroupLayout = device.createBindGroupLayout({
            label: "MC clamp args bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // indirectDraw
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, // dummy mcParams (not used but bound) OR skip? 
                // We reused binding 1 for MCParams in reset? No.
                // Shader expects: binding 0: indirectDraw, binding 2: maxVerts.
                // Wait, shader definition:
                // @group(0) @binding(0) var<storage, read_write> indirectDraw: IndirectDrawArgs;
                // @group(0) @binding(1) var<uniform> mcParams: MCParams; 
                // @group(0) @binding(2) var<uniform> maxVertsUniform: u32; 
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, // maxVerts
            ]
        });

        this.clampArgsPipeline = device.createComputePipeline({
            label: "MC clamp args pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [clampBindGroupLayout],
            }),
            compute: {
                module: device.createShaderModule({ code: mcClampArgsSrc }),
                entryPoint: "main",
            },
        });
        
        this.clampArgsBindGroup = device.createBindGroup({
            label: "MC clamp args bind group",
            layout: clampBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.indirectDrawBuffer } },
                { binding: 1, resource: { buffer: this.mcParamsBuffer } }, // unused but bound to satisfy layout if we kept it? 
                // Wait, I defined binding 1 in shader but said it might be unused. 
                // I must bind it if it's in shader.
                { binding: 2, resource: { buffer: this.maxVertsBuffer } },
            ]
        });
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
        
        // Reset indirect dispatch/draw counters using compute shader
        // This ensures correct ordering within the command stream
        const computePass = commandEncoder.beginComputePass({ label: "marching cubes compute pass" });
        computePass.setPipeline(this.resetPipeline);
        computePass.setBindGroup(0, this.resetBindGroup);
        computePass.dispatchWorkgroups(1);
        
        // 1. Calculate density grid (scatter particles to grid)
        // This is still dense/global because particles can be anywhere
        computePass.setPipeline(this.densityPipeline);
        computePass.setBindGroup(0, this.uniformsManager.bindGroup);
        computePass.setBindGroup(1, this.densityBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.mpmManager.nParticles / 256));
        
        // 2. List Active Blocks
        computePass.setPipeline(this.listBlocksPipeline);
        computePass.setBindGroup(0, this.listBlocksBindGroup);
        // Total blocks / 64 threads per group
        const [gx, gy, gz] = this.bufferManager.marchingCubesGridDims;
        const totalBlocks = Math.ceil(gx/8) * Math.ceil(gy/8) * Math.ceil(gz/8);
        computePass.dispatchWorkgroups(totalBlocks); // One workgroup per block now

        // 3. Calculate vertex densities and gradients ONLY for active blocks
        computePass.setPipeline(this.vertexDensityPipeline);
        computePass.setBindGroup(0, this.uniformsManager.bindGroup);
        computePass.setBindGroup(1, this.vertexDensityBindGroup);
        computePass.dispatchWorkgroupsIndirect(this.bufferManager.blockIndirectDispatchBuffer, 0);
        
        // 4. Generate mesh ONLY for active blocks
        computePass.setPipeline(this.generatePipeline);
        computePass.setBindGroup(0, this.uniformsManager.bindGroup);
        computePass.setBindGroup(1, this.generateBindGroup);
        computePass.dispatchWorkgroupsIndirect(this.bufferManager.blockIndirectDispatchBuffer, 0);

        // 5. Clamp indirect args to ensure we don't draw garbage
        computePass.setPipeline(this.clampArgsPipeline);
        computePass.setBindGroup(0, this.clampArgsBindGroup);
        computePass.dispatchWorkgroups(1);

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
