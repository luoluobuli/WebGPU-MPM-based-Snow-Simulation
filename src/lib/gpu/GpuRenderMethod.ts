export enum GpuRenderMethodType {
    Points,
    Volumetric,
    Ssfr,

}

export interface GpuRenderMethod {
    addDraw(renderPassEncoder: GPURenderPassEncoder): void;
}