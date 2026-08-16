import { describe, expect, it } from "vitest";

import {
    applyZubeiaWalkRootPitchInPlace,
    zubeiaWalkBodySettleOffsetM,
    ZUBEIA_WALK_BODY_SETTLE_DURATION_S,
    ZUBEIA_WALK_BODY_SETTLE_M,
    ZUBEIA_WALK_GROUND_SURFACE_Z_M,
    ZUBEIA_WALK_ROOT_PITCH_RAD,
    ZUBEIA_WALK_STRIDE_M,
} from "./ZubeiaWalkCalibration";

describe("ZubeiaWalkCalibration", () => {
    it("defines the measured root pitch, ground, and stride once", () => {
        expect(ZUBEIA_WALK_ROOT_PITCH_RAD).toBeCloseTo(
            -4.074 * Math.PI / 180,
            12,
        );
        expect(ZUBEIA_WALK_GROUND_SURFACE_Z_M).toBe(0.07);
        expect(ZUBEIA_WALK_STRIDE_M).toBe(0.54);
    });

    it("smoothly settles three cells after collision-free initialization", () => {
        expect(ZUBEIA_WALK_BODY_SETTLE_M).toBe(0.075);
        expect(ZUBEIA_WALK_BODY_SETTLE_DURATION_S).toBe(0.3);
        expect(zubeiaWalkBodySettleOffsetM(0)).toBeCloseTo(0);
        expect(zubeiaWalkBodySettleOffsetM(0.15)).toBeCloseTo(
            -0.0375,
        );
        expect(zubeiaWalkBodySettleOffsetM(0.3)).toBeCloseTo(
            -0.075,
        );
        expect(zubeiaWalkBodySettleOffsetM(3.6)).toBeCloseTo(-0.075);
    });

    it("rotates simulation y and z about x in place", () => {
        const values = new Float32Array([
            99, 99, 99,
            1, 3, 2,
        ]);

        expect(applyZubeiaWalkRootPitchInPlace(values, 3)).toBe(values);
        expect(values[3]).toBeCloseTo(1);
        expect(values[4]).toBeCloseTo(3.134509);
        expect(values[5]).toBeCloseTo(1.781812);
    });
});
