import { describe, expect, it } from "vitest";

import type {
    RigidPawSdfKinematicColliderAnimation,
} from "$lib/gpu/kinematicCollider/RigidPawSdfKinematicCollider";
import {
    buildZubeiaPawContactMask,
} from "./ZubeiaPawContactMask";

const identity = () => new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
]);

const boxTransform = (
    centerX: number,
    centerY: number,
    centerZ: number,
    halfExtentX = 0.49,
    halfExtentY = 0.49,
    halfExtentZ = 0.49,
) => new Float32Array([
    halfExtentX, 0, 0, 0,
    0, halfExtentY, 0, 0,
    0, 0, halfExtentZ, 0,
    centerX, centerY, centerZ, 1,
]);

const packedHalfValues = (bits: readonly number[]) => {
    const words = new Uint32Array(Math.ceil(bits.length / 2));
    bits.forEach((value, index) => {
        words[index >>> 1] |= index % 2 === 0
            ? value
            : value << 16;
    });

    return words;
};

const animationWith = ({
    packedPhi,
    sample,
}: {
    packedPhi: Uint32Array,
    sample: (timeS: number) => Float32Array<ArrayBuffer>,
}): RigidPawSdfKinematicColliderAnimation => ({
    primitiveCount: 1,
    sdfResolution: 2,
    packedPhi,
    sampleWorldFromUnitAtTimeS: (timeS, target) => {
        const sampled = sample(timeS);
        if (target === undefined) return sampled;

        target.set(sampled);
        return target;
    },
});

const allInsideSdf = () => packedHalfValues(
    Array.from({ length: 8 }, () => 0xbc00),
);

const buildMask = (
    animation: RigidPawSdfKinematicColliderAnimation,
    outputFrameIndex = 0,
) => buildZubeiaPawContactMask({
    animation,
    groundStripBounds: {
        min: [0, 0, -1],
        max: [7, 7, 0],
    },
    cellSizeM: 1,
    groundSurfaceZM: 0,
    outputFrameIndex,
    secondsPerFrame: 1,
});

describe("Zubeia swept paw contact mask", () => {
    it("samples the packed float16 field with x-fastest trilinear layout", () => {
        const xSplitSdf = packedHalfValues([
            0xbc00, 0x3c00,
            0xbc00, 0x3c00,
            0xbc00, 0x3c00,
            0xbc00, 0x3c00,
        ]);
        const animation = animationWith({
            packedPhi: xSplitSdf,
            sample: () => new Float32Array([
                1, 0, 0, 0,
                0, 0.5, 0, 0,
                0, 0, 1, 0,
                1, 0.5, 0, 1,
            ]),
        });
        const mask = buildZubeiaPawContactMask({
            animation,
            groundStripBounds: {
                min: [0, 0, -1],
                max: [2, 1, 0],
            },
            cellSizeM: 1,
            groundSurfaceZM: 0,
            outputFrameIndex: 0,
            secondsPerFrame: 1,
        });

        expect([...mask.contact]).toEqual([1, 0]);
        expect(mask.crossStripCellCount).toBe(2);
        expect(mask.walkingCellCount).toBe(1);
    });

    it("does not mark cells beneath an airborne paw", () => {
        const mask = buildMask(animationWith({
            packedPhi: allInsideSdf(),
            sample: () => boxTransform(3.5, 3.5, 2),
        }));

        expect(mask.contact.every(value => value === 0)).toBe(true);
        expect(mask.dilated.every(value => value === 0)).toBe(true);
        expect(mask.perimeter.every(value => value === 0)).toBe(true);
    });

    it("unions separated prints sampled at each output frame time", () => {
        const sampledTimes: number[] = [];
        const mask = buildMask(animationWith({
            packedPhi: allInsideSdf(),
            sample: timeS => {
                sampledTimes.push(timeS);
                return timeS === 0
                    ? boxTransform(0.5, 0.5, 0)
                    : boxTransform(4.5, 3.5, 0);
            },
        }), 1);

        expect(sampledTimes).toEqual([0, 1]);
        expect(mask.contact[0 * mask.walkingCellCount + 0]).toBe(1);
        expect(mask.contact[4 * mask.walkingCellCount + 3]).toBe(1);
        expect(
            mask.contact.reduce((sum, value) => sum + value, 0),
        ).toBe(2);
    });

    it("builds a bounded two-cell perimeter outside contact", () => {
        const centered = buildMask(animationWith({
            packedPhi: allInsideSdf(),
            sample: () => boxTransform(3.5, 3.5, 0),
        }));
        const corner = buildMask(animationWith({
            packedPhi: allInsideSdf(),
            sample: () => boxTransform(0.5, 0.5, 0),
        }));

        expect(
            centered.dilated.reduce((sum, value) => sum + value, 0),
        ).toBe(25);
        expect(
            centered.perimeter.reduce((sum, value) => sum + value, 0),
        ).toBe(24);
        expect(centered.perimeter[3 * 7 + 3]).toBe(0);
        expect(
            corner.dilated.reduce((sum, value) => sum + value, 0),
        ).toBe(9);
        expect(
            corner.perimeter.reduce((sum, value) => sum + value, 0),
        ).toBe(8);
    });

    it("accepts the animation's reusable pose target", () => {
        const fixedIdentity = identity();
        const animation = animationWith({
            packedPhi: allInsideSdf(),
            sample: () => fixedIdentity,
        });

        expect(() => buildMask(animation)).not.toThrow();
    });
});
