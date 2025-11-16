import { mat4, type Mat4 } from "wgpu-matrix";

export interface CameraControlScheme {
    viewMat(): Mat4;
    viewInvMat(): Mat4;
}

export interface CameraScreenDims {
    width(): number,
    height(): number,
}

export class Camera {
    private readonly controlScheme: CameraControlScheme;
    private readonly screenDims: CameraScreenDims;

    zNear = $state(0.01);
    zFar = $state(100);
    fov = $state(Math.PI / 2);

    readonly aspect = $derived.by(() => this.screenDims.width() / this.screenDims.height());

    readonly proj = $derived.by(() => mat4.perspective(this.fov, this.aspect, this.zNear, this.zFar));
    readonly viewMat = $derived.by(() => this.controlScheme.viewMat());
    readonly viewInvMat = $derived.by(() => this.controlScheme.viewInvMat());
    readonly viewProjMat = $derived.by(() => mat4.mul(this.proj, this.viewMat));
    readonly viewProjInvMat = $derived.by(() => mat4.inverse(this.viewProjMat));

    constructor({
        controlScheme,
        screenDims,
    }: {
        controlScheme: CameraControlScheme,
        screenDims: CameraScreenDims,
    }) {
        this.controlScheme = controlScheme;
        this.screenDims = screenDims;
    }
}