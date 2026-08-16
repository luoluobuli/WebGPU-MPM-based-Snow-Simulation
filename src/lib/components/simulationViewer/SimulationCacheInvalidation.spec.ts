import { beforeAll, describe, expect, it, vi } from "vitest";

import { GpuRenderMethodType } from "$lib/gpu/GpuRenderMethod";
import { GpuSimulationMethodType } from "$lib/gpu/GpuSimulationMethod";
import type { SimulationSceneConfig } from "./SimulationScene";
import {
    SIMULATION_FRAME_CACHE_SCHEMA_REVISION,
    SIMULATION_PHYSICS_SOURCE_FINGERPRINT,
} from "./SimulationPhysicsSourceFingerprint";

const kinematicGroundScene: SimulationSceneConfig = {
    spawnSource: {
        type: "zubeiaGround",
        profile: "preview",
    },
    colliderSource: {
        type: "zubeiaWalk",
        modelUrl: "/zubeia-walk-lod.glb",
        rigidPawSdfUrl: "/zubeia-paw-sdf.bin",
    },
    nParticles: 8,
    gridResolution: [16, 16, 16],
    renderMethodType: GpuRenderMethodType.Splats,
    simulationMethodType: GpuSimulationMethodType.FusedMlsMpm,
};

type SimulationStateConstructor =
    typeof import("./SimulationState.svelte").SimulationState;
type SimulationState = InstanceType<SimulationStateConstructor>;

let SimulationState: SimulationStateConstructor;

beforeAll(async () => {
    vi.doMock("./SimulationScene", () => ({
        defaultSimulationScene: kinematicGroundScene,
    }));
    ({ SimulationState } = await import("./SimulationState.svelte"));
});

const readCacheKey = (state: SimulationState) => (
    state as unknown as {
        currentTimelineCacheKey: () => string,
    }
).currentTimelineCacheKey();

const setLoadedContentFingerprint = (
    state: SimulationState,
    fingerprint: string,
) => {
    (
        state as unknown as {
            simulationContentFingerprint: string,
        }
    ).simulationContentFingerprint = fingerprint;
};

describe("simulation timeline cache invalidation", () => {
    it("changes namespace when loaded ground or collider bytes change", () => {
        const state = new SimulationState({
            scene: kinematicGroundScene,
            timeline: true,
        });
        setLoadedContentFingerprint(state, "ground-a:collider-a");
        const baselineKey = readCacheKey(state);

        setLoadedContentFingerprint(state, "ground-b:collider-a");
        const changedGroundKey = readCacheKey(state);

        setLoadedContentFingerprint(state, "ground-a:collider-b");
        const changedColliderKey = readCacheKey(state);

        expect(changedGroundKey).not.toBe(baselineKey);
        expect(changedColliderKey).not.toBe(baselineKey);
    });

    it("uses an explicit frame schema and a deterministic source fingerprint", () => {
        expect(SIMULATION_FRAME_CACHE_SCHEMA_REVISION).toBe(
            "position-material-v2",
        );
        expect(SIMULATION_PHYSICS_SOURCE_FINGERPRINT).toMatch(
            /^[0-9a-f]{16}$/,
        );
    });
});