import modelUrl from "$lib/assets/models/snow3.glb?url";
import colliderUrl from "$lib/assets/models/forest_scaled.glb?url";
import { GpuRenderMethodType } from "$lib/gpu/GpuRenderMethod";
import { GpuSimulationMethodType } from "$lib/gpu/GpuSimulationMethod";

export type SimulationSpawnSource =
    | {
        type: "mesh",
        url: string,
    }
    | {
        type: "proceduralForest",
        seed?: number,
    };

export type SimulationColliderSource =
    | {
        type: "mesh",
        url: string,
    }
    | null;

export type SimulationCameraDefaults = {
    radius?: number,
    lat?: number,
    long?: number,
    offset?: [number, number, number],
};

export type SimulationTimingConfig = {
    explicitMpmMaxSimulationTimestepS?: number,
    mlsMpmMaxSimulationTimestepS?: number,
    oneSimulationStepPerFrame?: boolean,
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
    renderMethodType: GpuRenderMethodType.Points,
    simulationMethodType: GpuSimulationMethodType.MlsMpm,
};

export const environmentScene: SimulationSceneConfig = {
    spawnSource: {
        type: "proceduralForest",
        seed: 0x5650f017,
    },
    colliderSource: null,
    nParticles: 300_000,
    gridResolution: [384, 384, 384],
    renderMethodType: GpuRenderMethodType.Splats,
    simulationMethodType: GpuSimulationMethodType.MlsMpm,
    camera: {
        radius: 8,
        lat: Math.PI / 4.6,
        long: Math.PI * 1.18,
        offset: [0, 0, -2.1],
    },
    timing: {
        mlsMpmMaxSimulationTimestepS: 1 / 256,
    },
};
