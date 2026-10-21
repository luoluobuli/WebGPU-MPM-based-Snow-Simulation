import {
    describe,
    expect,
    it,
} from "vitest";

import {
    GpuSimulationMethodType,
} from "$lib/gpu/GpuSimulationMethod";
import { GpuRenderMethodType } from "$lib/gpu/GpuRenderMethod";
import type {
    SimulationSceneConfig,
} from "./SimulationScene";
import {
    createPairedSimulationTimelineCacheKeys,
    createSimulationTimelineCacheKey,
    type SimulationTimelineCacheSettings,
} from "./SimulationTimelineCacheKey";


const canonicalColliderScene: SimulationSceneConfig = {
    spawnSource: {
        type: "zubeiaGround",
        profile: "final",
    },
    colliderSource: {
        type: "zubeiaWalk",
        modelUrl: "/models/zubeia.glb",
        rigidPawSdfUrl: "/models/zubeia-paws.sdfbin",
    },
    nParticles: 120_000,
    gridResolution: [128, 64, 64],
    renderMethodType: GpuRenderMethodType.Splats,
    simulationMethodType: GpuSimulationMethodType.Pbmpm,
    domain: {
        min: [-4, -2, -2],
        max: [4, 4, 2],
    },
    colliderFriction: 0.4,
    timeline: {
        frameCount: 73,
        secondsPerFrame: 1 / 24,
    },
    camera: {
        radius: 5,
        offset: [0, 1, 0],
    },
    renderCollider: true,
};

const settings: SimulationTimelineCacheSettings = {
    colliderFriction: 0.4,
    nParticles: 120_000,
    gridResolution: [128, 64, 64],
    simulationMethodType: GpuSimulationMethodType.Pbmpm,
    explicitMpmMaxSimulationTimestepS: 1 / 240,
    mlsMpmMaxSimulationTimestepS: 1 / 240,
    pbmpmMaxSimulationTimestepS: 1 / 240,
    pbmpmSolveIterations: 4,
    timelineFrameCount: 73,
    timelineSecondsPerFrame: 1 / 24,
};


describe("simulation timeline cache keys", () => {
    it("invalidates PBMPM frames when timestep or solve iterations change", () => {
        const keyFor = (overrides: Partial<SimulationTimelineCacheSettings>) => createSimulationTimelineCacheKey({
            scene: canonicalColliderScene,
            simulationContentFingerprint: "spawn-and-zubeia",
            settings: { ...settings, ...overrides },
        });

        expect(keyFor({ pbmpmSolveIterations: 8 })).not.toBe(keyFor({}));
        expect(keyFor({ pbmpmMaxSimulationTimestepS: 1 / 60 })).not.toBe(keyFor({}));
    });

    it("cross-matches actual and counterpart keys in both page roles", () => {
        const colliderPageKeys = createPairedSimulationTimelineCacheKeys({
            canonicalColliderScene,
            colliderSimulationContentFingerprint: "spawn-and-zubeia",
            controlSimulationContentFingerprint: "spawn-and-null",
            activeHasCollider: true,
            settings,
        });
        const controlPageKeys = createPairedSimulationTimelineCacheKeys({
            canonicalColliderScene,
            colliderSimulationContentFingerprint: "spawn-and-zubeia",
            controlSimulationContentFingerprint: "spawn-and-null",
            activeHasCollider: false,
            settings,
        });

        expect(colliderPageKeys.activeCacheKey).toBe(
            controlPageKeys.counterpartCacheKey,
        );
        expect(controlPageKeys.activeCacheKey).toBe(
            colliderPageKeys.counterpartCacheKey,
        );
        expect(controlPageKeys.retainedCacheKeys).toEqual(
            colliderPageKeys.retainedCacheKeys,
        );
        expect(new Set(colliderPageKeys.retainedCacheKeys).size).toBe(2);
    });

    it("ignores presentation fields but includes active physics settings", () => {
        const baseKey = createSimulationTimelineCacheKey({
            scene: canonicalColliderScene,
            simulationContentFingerprint: "spawn-and-zubeia",
            settings,
        });
        const presentationOnlyKey = createSimulationTimelineCacheKey({
            scene: {
                ...canonicalColliderScene,
                camera: {
                    radius: 99,
                    offset: [1, 1, 1],
                },
                renderCollider: false,
            },
            simulationContentFingerprint: "spawn-and-zubeia",
            settings,
        });
        const changedFrictionKey = createSimulationTimelineCacheKey({
            scene: canonicalColliderScene,
            simulationContentFingerprint: "spawn-and-zubeia",
            settings: {
                ...settings,
                colliderFriction: settings.colliderFriction + 0.1,
            },
        });
        const changedContentKey = createSimulationTimelineCacheKey({
            scene: canonicalColliderScene,
            simulationContentFingerprint: "different-zubeia-bytes",
            settings,
        });

        expect(presentationOnlyKey).toBe(baseKey);
        expect(changedFrictionKey).not.toBe(baseKey);
        expect(changedContentKey).not.toBe(baseKey);
    });
});
