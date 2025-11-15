import { mat4, type Mat4 } from "wgpu-matrix";

export interface CameraControlScheme {
    viewTransform(): Mat4,
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
    readonly viewInv = $derived.by(() => this.controlScheme.viewTransform());
    readonly viewInvProj = $derived.by(() => mat4.mul(this.proj, this.viewInv));

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