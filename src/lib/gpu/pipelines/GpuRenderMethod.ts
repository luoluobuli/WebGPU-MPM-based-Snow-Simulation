export enum GpuRenderMethodType {
    Points,
    Raymarch,
}

export interface GpuRenderMethod {
    addDraw(renderPassEncoder: GPURenderPassEncoder): void;
}