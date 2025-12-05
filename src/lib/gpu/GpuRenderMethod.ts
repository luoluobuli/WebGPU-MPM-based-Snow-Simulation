export enum GpuRenderMethodType {
    Points,
    Volumetric,
    Ssfr,
    MarchingCubes,
}

export interface GpuRenderMethod {
    addDraw(renderPassEncoder: GPURenderPassEncoder): void;
}