import { describe, expect, it } from "vitest";

import {
    packRigidPawSdfColliderPose,
    rigidPawSdfPoseBufferByteLength,
    RIGID_PAW_SDF_POSE_BYTES,
} from "./GpuRigidPawSdfKinematicColliderManager";

const translation = (
    x: number,
    y: number,
    z: number,
) => new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1,
]);

describe("rigid paw SDF pose packing", () => {
    it("allocates one complete pose struct when collision is disabled", () => {
        expect(rigidPawSdfPoseBufferByteLength(0)).toBe(
            RIGID_PAW_SDF_POSE_BYTES,
        );
        expect(rigidPawSdfPoseBufferByteLength(4)).toBe(
            4 * RIGID_PAW_SDF_POSE_BYTES,
        );
    });

    it("packs current, inverse, previous, and interval duration", () => {
        const target = new Uint8Array(RIGID_PAW_SDF_POSE_BYTES);

        packRigidPawSdfColliderPose({
            currentWorldFromUnit: translation(3, 4, 5),
            previousWorldFromUnit: translation(1, 2, 3),
            timestepS: 0.125,
            primitiveCount: 1,
            target,
        });

        const values = new Float32Array(target.buffer);
        expect([...values.slice(12, 16)]).toEqual([3, 4, 5, 1]);
        expect([...values.slice(28, 32)]).toEqual([-3, -4, -5, 1]);
        expect([...values.slice(44, 48)]).toEqual([1, 2, 3, 1]);
        expect(values[48]).toBe(0.125);
    });

    it("rejects non-increasing intervals and malformed targets", () => {
        expect(() => packRigidPawSdfColliderPose({
            currentWorldFromUnit: translation(0, 0, 0),
            previousWorldFromUnit: translation(0, 0, 0),
            timestepS: 0,
            primitiveCount: 1,
            target: new Uint8Array(RIGID_PAW_SDF_POSE_BYTES),
        })).toThrow("timestep must be positive");

        expect(() => packRigidPawSdfColliderPose({
            currentWorldFromUnit: translation(0, 0, 0),
            previousWorldFromUnit: translation(0, 0, 0),
            timestepS: 1,
            primitiveCount: 1,
            target: new Uint8Array(4),
        })).toThrow("208 aligned bytes");
    });
});
