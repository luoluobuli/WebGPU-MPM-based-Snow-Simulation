import { afterEach, describe, expect, it, vi } from "vitest";
import {
    fingerprintLoadedSpawnSource,
    loadAssetContentFingerprint,
    loadZubeiaColliderContentFingerprint,
} from "./SimulationContentFingerprint";

describe("loaded simulation content fingerprints", () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it("changes when generated ground points change", () => {
        const baseline = {
            type: "points" as const,
            points: new Float32Array([0, 1, 2, 4]),
        };
        const changed = {
            type: "points" as const,
            points: new Float32Array([0, 1, 2, 5]),
        };

        expect(fingerprintLoadedSpawnSource(changed)).not.toBe(
            fingerprintLoadedSpawnSource(baseline),
        );
    });

    it("hashes the bytes returned for a collider asset", async () => {
        const fetchMock = vi.spyOn(globalThis, "fetch")
            .mockResolvedValue(new Response(
                new Uint8Array([1, 2, 3, 4]),
            ));

        const fingerprint = await loadAssetContentFingerprint(
            "/zubeia-walk-paw-sdf.bin",
        );

        expect(fingerprint).toHaveLength(16);
        expect(fetchMock).toHaveBeenCalledWith(
            "/zubeia-walk-paw-sdf.bin",
        );
    });

    it("invalidates Zubeia collider content when either asset changes", async () => {
        const modelUrl = "/zubeia-walk-lod.glb";
        const rigidPawSdfUrl = "/zubeia-paw-sdf.bin";
        const assetBytes = new Map<string, Uint8Array>([
            [modelUrl, new Uint8Array([1, 2, 3, 4])],
            [rigidPawSdfUrl, new Uint8Array([5, 6, 7, 8])],
        ]);
        const fetchMock = vi.spyOn(globalThis, "fetch")
            .mockImplementation(async input => {
                const bytes = assetBytes.get(String(input));
                if (bytes === undefined) {
                    return new Response(null, { status: 404 });
                }

                const body = new ArrayBuffer(bytes.byteLength);
                new Uint8Array(body).set(bytes);

                return new Response(body);
            });
        const fingerprint = async () =>
            await loadZubeiaColliderContentFingerprint({
                modelUrl,
                rigidPawSdfUrl,
            });

        const baseline = await fingerprint();
        assetBytes.set(
            modelUrl,
            new Uint8Array([1, 2, 3, 9]),
        );
        const changedModel = await fingerprint();
        assetBytes.set(
            modelUrl,
            new Uint8Array([1, 2, 3, 4]),
        );
        assetBytes.set(
            rigidPawSdfUrl,
            new Uint8Array([5, 6, 7, 9]),
        );
        const changedSdf = await fingerprint();

        expect(changedModel).not.toBe(baseline);
        expect(changedSdf).not.toBe(baseline);
        expect(fetchMock).toHaveBeenCalledWith(modelUrl);
        expect(fetchMock).toHaveBeenCalledWith(rigidPawSdfUrl);
    });
});
