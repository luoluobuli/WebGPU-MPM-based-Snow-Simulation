
export class GpuVolumetricBufferManager {
    massGridBuffer: GPUBuffer;
    outputTexture: GPUTexture;
    outputTextureView: GPUTextureView;
    depthTexture: GPUTexture;
    depthTextureView: GPUTextureView;
    readonly gridResolution: [number, number, number];

    constructor({
        device,
        gridResolutionX,
        gridResolutionY,
        gridResolutionZ,
        screenDims,
    }: {
        device: GPUDevice,
        gridResolutionX: number,
        gridResolutionY: number,
        gridResolutionZ: number,
        screenDims: { width: number, height: number },
    }) {
        this.gridResolution = [gridResolutionX, gridResolutionY, gridResolutionZ];

        this.massGridBuffer = device.createBuffer({
            label: "volumetric mass grid buffer",
            size: gridResolutionX * gridResolutionY * gridResolutionZ * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        this.outputTexture = device.createTexture({
            label: "volumetric output texture",
            size: [screenDims.width, screenDims.height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this.outputTextureView = this.outputTexture.createView();

        this.depthTexture = device.createTexture({
            label: "volumetric depth texture",
            size: [screenDims.width, screenDims.height],
            format: "r32float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this.depthTextureView = this.depthTexture.createView();
    }

    resize(device: GPUDevice, width: number, height: number) {
        this.outputTexture.destroy();
        this.depthTexture.destroy();
        
        this.outputTexture = device.createTexture({
            label: "volumetric output texture",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this.outputTextureView = this.outputTexture.createView();

        this.depthTexture = device.createTexture({
            label: "volumetric depth texture",
            size: [width, height],
            format: "r32float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this.depthTextureView = this.depthTexture.createView();
    }

    destroy() {
        this.massGridBuffer.destroy();
        this.outputTexture.destroy();
        this.depthTexture.destroy();
    }
}
