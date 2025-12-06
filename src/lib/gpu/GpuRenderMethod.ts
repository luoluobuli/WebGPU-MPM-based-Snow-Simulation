export enum GpuRenderMethodType {
    Points,
    Volumetric,
    Ssfr,
    MarchingCubes,
}

export interface GpuRenderMethod {
    nPrerenderPasses(): number;
    addPrerenderPasses(commandEncoder: GPUCommandEncoder, depthTextureView: GPUTextureView): void;
    addFinalDraw(renderPassEncoder: GPURenderPassEncoder): void;
    resize(device: GPUDevice, width: number, height: number, depthTextureView: GPUTextureView): void;
    destroy(): void;
}