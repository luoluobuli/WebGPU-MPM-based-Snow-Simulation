import { mat4, vec3, type Mat4 } from "wgpu-matrix";
import type { CameraControlScheme } from "./Camera.svelte";

export class CameraOrbit implements CameraControlScheme {
    radius = $state(2);
    lat = $state(Math.PI / 6);
    long = $state(Math.PI / 4);
    
    offset = $state(vec3.zero());

    readonly basePos = $derived(vec3.fromValues(
        this.radius * Math.cos(this.long) * Math.cos(this.lat),
        this.radius * Math.cos(this.long) * Math.sin(this.lat),
        this.radius * Math.sin(this.long),
    ));

    viewTransform(): Mat4 {
        const eye = vec3.add(this.basePos, this.offset);
        const target = this.offset;
        const up = vec3.fromValues(0, 0, 1);
        return mat4.lookAt(eye, target, up);
    }
}