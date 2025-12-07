export enum GpuRenderMethodType {
    Points,
    Volumetric,
    Ssfr,
    MarchingCubes,
}

export interface GpuRenderMethod {
    prerenderPasses(): string[];
    addPrerenderPasses(commandEncoder: GPUCommandEncoder, depthTextureView: GPUTextureView): void;
    addFinalDraw(renderPassEncoder: GPURenderPassEncoder): void;
    resize(device: GPUDevice, width: number, height: number, depthTextureView: GPUTextureView): void;
    destroy(): void;
}