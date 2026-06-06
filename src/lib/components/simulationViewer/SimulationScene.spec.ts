import { describe, expect, it } from "vitest";

import { defaultSimulationScene, environmentScene } from "./SimulationScene";
import { GRAVITATIONAL_ACCELERATION_M_PER_S2 } from "$lib/gpu/gravity";
import { GpuRenderMethodType } from "$lib/gpu/GpuRenderMethod";

describe("SimulationScene timing", () => {
    it("keeps the default scene at the authored high-frequency timestep", () => {
        expect(GRAVITATIONAL_ACCELERATION_M_PER_S2).toBe(9.81);
        expect(defaultSimulationScene.timing?.mlsMpmMaxSimulationTimestepS).toBe(1 / 1024);
        expect(defaultSimulationScene.timing?.oneSimulationStepPerFrame).toBe(true);
        expect(defaultSimulationScene.renderMethodType).toBe(GpuRenderMethodType.Splats);
    });

    it("keeps the environment scene collider-free with grid support matched to sparse generated particles", () => {
        expect(environmentScene.colliderSource).toBeNull();
        expect(environmentScene.nParticles).toBe(220_000);
        expect(environmentScene.gridResolution).toEqual([128, 128, 128]);
        expect(environmentScene.timing?.mlsMpmMaxSimulationTimestepS).toBe(1 / 1024);
    });
});
