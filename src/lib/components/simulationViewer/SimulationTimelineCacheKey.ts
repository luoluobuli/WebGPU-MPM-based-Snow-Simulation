import {
    SIMULATION_FRAME_CACHE_SCHEMA_REVISION,
    SIMULATION_PHYSICS_SOURCE_FINGERPRINT,
} from "./SimulationPhysicsSourceFingerprint";
import { fingerprintString } from "./SimulationCacheFingerprint";
import type {
    SimulationSceneConfig,
} from "./SimulationScene";


export type SimulationTimelineCacheSettings = {
    colliderFriction: number,
    nParticles: number,
    gridResolution: readonly [number, number, number],
    simulationMethodType: SimulationSceneConfig["simulationMethodType"],
    explicitMpmMaxSimulationTimestepS: number,
    mlsMpmMaxSimulationTimestepS: number,
    timelineFrameCount: number,
    timelineSecondsPerFrame: number,
};

type SimulationTimelineCacheKeyArgs = {
    scene: SimulationSceneConfig,
    simulationContentFingerprint: string,
    settings: SimulationTimelineCacheSettings,
};

type PairedSimulationTimelineCacheKeyArgs = {
    canonicalColliderScene: SimulationSceneConfig,
    colliderSimulationContentFingerprint: string,
    controlSimulationContentFingerprint: string,
    activeHasCollider: boolean,
    settings: SimulationTimelineCacheSettings,
};

export type PairedSimulationTimelineCacheKeys = {
    activeCacheKey: string,
    counterpartCacheKey: string,
    colliderCacheKey: string,
    controlCacheKey: string,
    retainedCacheKeys: readonly [string, string],
};


export const createSimulationTimelineCacheKey = ({
    scene,
    simulationContentFingerprint,
    settings,
}: SimulationTimelineCacheKeyArgs) => fingerprintString(JSON.stringify({
    simulationScene: {
        spawnSource: scene.spawnSource,
        colliderSource: scene.colliderSource,
        domain: scene.domain,
        colliderFriction: settings.colliderFriction,
    },
    frameCacheSchemaRevision: SIMULATION_FRAME_CACHE_SCHEMA_REVISION,
    physicsSourceFingerprint: SIMULATION_PHYSICS_SOURCE_FINGERPRINT,
    simulationContentFingerprint,
    nParticles: settings.nParticles,
    gridResolution: settings.gridResolution,
    simulationMethodType: settings.simulationMethodType,
    explicitMpmMaxSimulationTimestepS:
        settings.explicitMpmMaxSimulationTimestepS,
    mlsMpmMaxSimulationTimestepS:
        settings.mlsMpmMaxSimulationTimestepS,
    timelineFrameCount: settings.timelineFrameCount,
    timelineSecondsPerFrame: settings.timelineSecondsPerFrame,
}));

export const createPairedSimulationTimelineCacheKeys = ({
    canonicalColliderScene,
    colliderSimulationContentFingerprint,
    controlSimulationContentFingerprint,
    activeHasCollider,
    settings,
}: PairedSimulationTimelineCacheKeyArgs): PairedSimulationTimelineCacheKeys => {
    const colliderCacheKey = createSimulationTimelineCacheKey({
        scene: canonicalColliderScene,
        simulationContentFingerprint: colliderSimulationContentFingerprint,
        settings,
    });
    const controlCacheKey = createSimulationTimelineCacheKey({
        scene: {
            ...canonicalColliderScene,
            colliderSource: null,
        },
        simulationContentFingerprint: controlSimulationContentFingerprint,
        settings,
    });

    return {
        activeCacheKey: activeHasCollider
            ? colliderCacheKey
            : controlCacheKey,
        counterpartCacheKey: activeHasCollider
            ? controlCacheKey
            : colliderCacheKey,
        colliderCacheKey,
        controlCacheKey,
        retainedCacheKeys: [colliderCacheKey, controlCacheKey],
    };
};
