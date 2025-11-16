import { mat3, mat4, vec3, type Mat4 } from "wgpu-matrix";
import type { CameraControlScheme } from "./Camera.svelte";
import { mod, PI, PI_2, REV } from "./util";

type Point = { x: number; y: number };


const ORBIT_CONTROL_SCALE = 0.005;

export class CameraOrbit implements CameraControlScheme {
    radius = $state(4);
    lat = $state(PI / 6);
    long = $state(11 * PI / 8);
    
    offset = $state(vec3.zero());

    readonly orientation = $derived(mat4.mul(
        mat4.rotationZ(-this.long),
        mat4.rotationX(-this.lat + PI_2),
    ));

    readonly pos = $derived(vec3.add(
        vec3.transformMat4(vec3.fromValues(0, 0, this.radius), this.orientation),
        this.offset,
    ));
    readonly rot = $derived(this.orientation);

    readonly viewInv = $derived(mat4.mul(mat4.translation(this.pos), this.rot));
    readonly view = $derived(mat4.inverse(this.viewInv));

    viewMat(): Mat4 {
        return this.view;
    }

    viewInvMat(): Mat4 {
        return this.viewInv;
    }

    pan(movement: Point) {
        this.offset = vec3.add(
            this.offset,
            vec3.transformMat3(
                vec3.fromValues(-movement.x * ORBIT_CONTROL_SCALE, movement.y * ORBIT_CONTROL_SCALE, 0),
                mat3.fromMat4(this.orientation),
            ),
        );
    }

    turn(movement: Point) {
        this.lat = mod(this.lat + movement.y * ORBIT_CONTROL_SCALE, REV);

        if (PI_2 < this.lat && this.lat < 3 * PI_2) {
            this.long = mod(this.long - movement.x * ORBIT_CONTROL_SCALE, REV);
        } else {
            this.long = mod(this.long + movement.x * ORBIT_CONTROL_SCALE, REV);
        }
    }
}