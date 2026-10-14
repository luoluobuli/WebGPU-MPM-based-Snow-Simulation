import modelUrl from "$lib/assets/models/snow3.glb?url";
import colliderUrl from "$lib/assets/models/forest_scaled.glb?url";
import treeModelUrl from "$lib/assets/models/tree0.glb?url";
import zubeiaWalkModelUrl from "$lib/assets/models/zubeia-walk-lod.glb?url";
import zubeiaPawSdfUrl from "$lib/assets/models/zubeia-paw-sdf.bin?url";
import { GpuRenderMethodType } from "$lib/gpu/GpuRenderMethod";
import { GpuSimulationMethodType } from "$lib/gpu/GpuSimulationMethod";
import {
    ZUBEIA_GROUND_DOMAIN_BOUNDS,
    ZUBEIA_GROUND_FINAL_PARTICLE_COUNT,
    ZUBEIA_GROUND_GRID_RESOLUTION,
    ZUBEIA_GROUND_PREVIEW_PARTICLE_COUNT,
    type ZubeiaGroundProfile,
} from "./zubeia/ZubeiaGround";

export type SimulationSpawnSource =
    | {
        type: "mesh",
        url: string,
    }
    | {
        type: "treeModel",
        url: string,
    }
    | {
        type: "zubeiaGround",
        profile: ZubeiaGroundProfile,
    };

export type SimulationColliderSource =
    | {
        type: "mesh",
        url: string,
    }
    | {
        type: "zubeiaWalk",
        modelUrl: string,
        rigidPawSdfUrl: string,
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

export type SimulationTimelineConfig = {
    frameCount: number,
    secondsPerFrame: number,
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
    timeline?: SimulationTimelineConfig,
    domain?: SimulationDomain,
    colliderFriction?: number,
    renderCollider?: boolean,
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

export const environmentScene: SimulationSceneConfig = {
    spawnSource: {
        type: "treeModel",
        url: treeModelUrl,
    },
    colliderSource: null,
    nParticles: 480_000,
    gridResolution: [128, 128, 128],
    renderMethodType: GpuRenderMethodType.Splats,
    simulationMethodType: GpuSimulationMethodType.MlsMpm,
    camera: {
        radius: 8,
        lat: Math.PI / 4.6,
        long: Math.PI * 1.18,
        offset: [0, 0, 0],
    },
    timing: {
        mlsMpmMaxSimulationTimestepS: 1 / 1024,
    },
};

const createZubeiaGroundScene = ({
    profile,
    animated,
}: {
    profile: ZubeiaGroundProfile,
    animated: boolean,
}): SimulationSceneConfig => ({
    spawnSource: {
        type: "zubeiaGround",
        profile,
    },
    colliderSource: animated
        ? {
            type: "zubeiaWalk",
            modelUrl: zubeiaWalkModelUrl,
            rigidPawSdfUrl: zubeiaPawSdfUrl,
        }
        : null,
    nParticles: profile === "final"
        ? ZUBEIA_GROUND_FINAL_PARTICLE_COUNT
        : ZUBEIA_GROUND_PREVIEW_PARTICLE_COUNT,
    gridResolution: [...ZUBEIA_GROUND_GRID_RESOLUTION],
    renderMethodType: GpuRenderMethodType.Splats,
    simulationMethodType: GpuSimulationMethodType.FusedMlsMpm,
    camera: {
        radius: 9,
        lat: 0.75,
        long: Math.PI * 0.5,
        offset: [0, 0.9, 0.9],
    },
    timing: {
        mlsMpmMaxSimulationTimestepS: 1 / 240,
        oneSimulationStepPerFrame: true,
    },
    timeline: {
        frameCount: 109,
        secondsPerFrame: 1 / 30,
    },
    domain: {
        min: [...ZUBEIA_GROUND_DOMAIN_BOUNDS.min],
        max: [...ZUBEIA_GROUND_DOMAIN_BOUNDS.max],
    },
    colliderFriction: 0.65,
});

export const groundScene = createZubeiaGroundScene({
    profile: "preview",
    animated: false,
});

export const kinematicGroundScene = createZubeiaGroundScene({
    profile: "preview",
    animated: true,
});

export const kinematicGroundFinalScene = createZubeiaGroundScene({
    profile: "final",
    animated: true,
});
