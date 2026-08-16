import { describe, expect, it } from "vitest";
import {
    combineCacheFingerprints,
    fingerprintBytes,
    fingerprintFloat32Values,
    fingerprintString,
} from "./SimulationCacheFingerprint";

describe("simulation cache fingerprints", () => {
    it("is stable for identical source content", () => {
        expect(fingerprintString("solver source")).toBe(
            fingerprintString("solver source"),
        );
        expect(fingerprintFloat32Values([0.1, 0.2, 0.3])).toBe(
            fingerprintFloat32Values(new Float32Array([0.1, 0.2, 0.3])),
        );
    });

    it("changes when one ground or collider byte changes", () => {
        const baseline = new Uint8Array([1, 2, 3, 4]);
        const changed = new Uint8Array([1, 2, 3, 5]);

        expect(fingerprintBytes(changed)).not.toBe(
            fingerprintBytes(baseline),
        );
    });

    it("keeps component ordering in the combined identity", () => {
        const ground = fingerprintString("ground");
        const collider = fingerprintString("collider");

        expect(combineCacheFingerprints([ground, collider])).not.toBe(
            combineCacheFingerprints([collider, ground]),
        );
    });
});
