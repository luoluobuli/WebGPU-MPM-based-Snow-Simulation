import modelUrl from "$lib/assets/models/snow3.glb?url";
import colliderUrl from "$lib/assets/models/forest_scaled.glb?url";
import { GpuRenderMethodType } from "$lib/gpu/GpuRenderMethod";
import { GpuSimulationMethodType } from "$lib/gpu/GpuSimulationMethod";

export type SimulationSpawnSource = {
    type: "mesh",
    url: string,
};

export type SimulationColliderSource = {
    type: "mesh",
    url: string,
} | null;

export type SimulationCameraDefaults = {
    radius?: number,
    lat?: number,
    long?: number,
    offset?: [number, number, number],
};

export type SimulationTimingConfig = {
    explicitMpmMaxSimulationTimestepS?: number,
    mlsMpmMaxSimulationTimestepS?: number,
    pbmpmMaxSimulationTimestepS?: number,
    pbmpmSolveIterations?: number,
    oneSimulationStepPerFrame?: boolean,
};

export type SimulationDomain = {
    min: [number, number, number],
    max: [number, number, number],
};

export type SimulationSceneConfig = {
    spawnSource: SimulationSpawnSource,
    colliderSource: SimulationColliderSource,
    nParticles: number,
    gridResolution: [number, number, number],
    renderMethodType: GpuRenderMethodType,
    simulationMethodType: GpuSimulationMethodType,
    camera?: SimulationCameraDefaults,
    timing?: SimulationTimingConfig,
    domain?: SimulationDomain,
    colliderFriction?: number,
};

export const defaultSimulationScene: SimulationSceneConfig = {
    spawnSource: {
        type: "mesh",
        url: modelUrl,
    },
    colliderSource: {
        type: "mesh",
        url: colliderUrl,
    },
    nParticles: 300_000,
    gridResolution: [384, 384, 384],
    renderMethodType: GpuRenderMethodType.Splats,
    simulationMethodType: GpuSimulationMethodType.MlsMpm,
    timing: {
        mlsMpmMaxSimulationTimestepS: 1 / 1024,
        oneSimulationStepPerFrame: true,
    },
};
