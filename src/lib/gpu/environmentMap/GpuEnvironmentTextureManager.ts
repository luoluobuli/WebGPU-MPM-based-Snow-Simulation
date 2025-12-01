export class GpuEnvironmentTextureManager {
    readonly environmentTexture: GPUTexture;
    
    constructor({
        device,
        imageBitmap,
    }: {
        device: GPUDevice,
        imageBitmap: ImageBitmap,
    }) {
        const environmentTexture = device.createTexture({
            label: "environment texture",
            size: {
                width: imageBitmap.width,
                height: imageBitmap.height,
                depthOrArrayLayers: 1,
            },
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });

        device.queue.copyExternalImageToTexture(
            {
                source: imageBitmap,
            },
            {
                texture: environmentTexture,
            },
            [imageBitmap.width, imageBitmap.height, 1],
        );

        this.environmentTexture = environmentTexture;
    }
}