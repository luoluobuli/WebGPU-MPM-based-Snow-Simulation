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
    pbmpmMaxSimulationTimestepS: number,
    pbmpmSolveIterations: number,
    timelineFrameCount: number,
    timelineSecondsPerFrame: number,
};

type SimulationTimelineCacheKeyArgs = {
    scene: SimulationSceneConfig,
    simulationContentFingerprint: string,
    settings: SimulationTimelineCacheSettings,
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
    pbmpmMaxSimulationTimestepS: settings.pbmpmMaxSimulationTimestepS,
    pbmpmSolveIterations: settings.pbmpmSolveIterations,
    timelineFrameCount: settings.timelineFrameCount,
    timelineSecondsPerFrame: settings.timelineSecondsPerFrame,
}));
