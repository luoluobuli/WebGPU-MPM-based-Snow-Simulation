export enum GpuRenderMethodType {
    Points,
    Volumetric,
}

export interface GpuRenderMethod {
    addDraw(renderPassEncoder: GPURenderPassEncoder): void;
}