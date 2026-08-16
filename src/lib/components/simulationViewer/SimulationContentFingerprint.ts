import type { SpawnPointSource } from "$lib/gpu/particleInitialize/GpuSpawnVolumeBufferManager";
import {
    combineCacheFingerprints,
    fingerprintBytes,
    fingerprintString,
} from "./SimulationCacheFingerprint";

export const fingerprintLoadedSpawnSource = (
    source: SpawnPointSource,
) => {
    if (source.type === "points") {
        return combineCacheFingerprints([
            fingerprintString(source.type),
            fingerprintBytes(source.points),
        ]);
    }

    // Mesh-spawn timelines are outside the Zubeia proof route. Keep their
    // identity deterministic without allocating a second flattened mesh.
    return fingerprintString(JSON.stringify({
        type: source.type,
        vertices: source.vertices,
        objects: source.objects,
    }));
};

export const loadAssetContentFingerprint = async (
    url: string,
) => {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(
            `could not fingerprint simulation asset ${url}: `
            + `${response.status} ${response.statusText}`,
        );
    }

    return fingerprintBytes(await response.arrayBuffer());
};

export const loadZubeiaColliderContentFingerprint = async ({
    modelUrl,
    rigidPawSdfUrl,
}: {
    modelUrl: string,
    rigidPawSdfUrl: string,
}) => combineCacheFingerprints(
    await Promise.all(
        [modelUrl, rigidPawSdfUrl].map(loadAssetContentFingerprint),
    ),
);
