import { describe, expect, it } from "vitest";

import {
    encodeRigidPawSdfColliderAsset,
    parseRigidPawSdfColliderAsset,
    type RigidPawSdfColliderAssetSource,
} from "./rigidPawSdfColliderAsset";

const source = (): RigidPawSdfColliderAssetSource => ({
    cycleCount: 3,
    sdfResolution: 4,
    sampleRateHz: 2,
    cycleDurationS: 1,
    durationS: 3,
    stridePerCycle: [0, 0.54, 0],
    volumes: [
        {
            minReference: [-1, -2, -3],
            maxReference: [1, 2, 3],
            referenceFrameIndex: 0,
        },
        {
            minReference: [4, 5, 6],
            maxReference: [7, 8, 9],
            referenceFrameIndex: 1,
        },
    ],
    verticalCorrectionM: new Float32Array([
        0, 10,
        2, 12,
        4, 14,
    ]),
    packedPhi: new Uint32Array(64),
});

describe("rigid paw SDF collider asset", () => {
    it("round-trips static fields, stance bounds, and motion correction", () => {
        const parsed = parseRigidPawSdfColliderAsset(
            encodeRigidPawSdfColliderAsset(source()).buffer,
        );

        expect(parsed.volumeCount).toBe(2);
        expect(parsed.motionFrameCount).toBe(3);
        expect(parsed.sdfResolution).toBe(4);
        expect(parsed.volumes[1]).toEqual({
            minReference: [4, 5, 6],
            maxReference: [7, 8, 9],
            referenceFrameIndex: 1,
        });
        expect([
            ...parsed.sampleVerticalCorrectionAtGaitTimeS(0.25),
        ]).toEqual([1, 11]);
        expect([
            ...parsed.sampleVerticalCorrectionAtGaitTimeS(1),
        ]).toEqual([4, 14]);
    });

    it("rejects a field payload with the wrong packed size", () => {
        expect(() => encodeRigidPawSdfColliderAsset({
            ...source(),
            packedPhi: new Uint32Array(1),
        })).toThrow("packedPhi must contain 64 words");
    });

    it("rejects corrupt offsets instead of accepting ambiguous data", () => {
        const bytes = encodeRigidPawSdfColliderAsset(source());
        new DataView(bytes.buffer).setUint32(48, 0, true);

        expect(() => parseRigidPawSdfColliderAsset(
            bytes.buffer,
        )).toThrow("sections are not tightly packed");
    });
});
