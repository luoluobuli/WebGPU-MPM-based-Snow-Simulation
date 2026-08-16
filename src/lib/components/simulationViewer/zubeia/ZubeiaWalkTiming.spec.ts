import { describe, expect, it } from "vitest";

import {
    sampleZubeiaWalkTiming,
    ZUBEIA_WALK_ANIMATION_NAME,
} from "./ZubeiaWalkTiming";

const timingAt = (timeS: number) => sampleZubeiaWalkTiming({
    timeS,
    cycleDurationS: 1.2,
    durationS: 3.6,
    stridePerCycle: [0, 0.54, 0],
});

describe("sampleZubeiaWalkTiming", () => {
    it("uses the authored clip name", () => {
        expect(ZUBEIA_WALK_ANIMATION_NAME).toBe("walk_forward");
    });

    it("loops the in-place gait while advancing continuous root motion", () => {
        expect(timingAt(1.8).gaitTimeS).toBeCloseTo(0.6);
        expect(timingAt(1.8).rootTranslation).toEqual([
            0,
            0.81,
            0,
        ]);
    });

    it("clamps before the first pose and after three complete cycles", () => {
        expect(timingAt(-1)).toEqual({
            gaitTimeS: 0,
            rootTranslation: [0, 0, 0],
        });
        expect(timingAt(4)).toEqual({
            gaitTimeS: 1.2,
            rootTranslation: [0, 1.62, 0],
        });
    });
});
