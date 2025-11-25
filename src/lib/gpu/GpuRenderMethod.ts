export enum GpuRenderMethodType {
    Points,
    Raymarch,
    Volumetric,
}

export interface GpuRenderMethod {
    addDraw(renderPassEncoder: GPURenderPassEncoder): void;
}