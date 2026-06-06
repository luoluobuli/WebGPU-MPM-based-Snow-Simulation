import { describe, expect, it } from "vitest";

import { defaultSimulationScene, environmentScene } from "./SimulationScene";
import { GRAVITATIONAL_ACCELERATION_M_PER_S2 } from "$lib/gpu/gravity";

describe("SimulationScene timing", () => {
    it("keeps the default scene at normal gravity and default timing", () => {
        expect(GRAVITATIONAL_ACCELERATION_M_PER_S2).toBe(9.81);
        expect(defaultSimulationScene.timing?.mlsMpmMaxSimulationTimestepS).toBeUndefined();
    });

    it("uses a larger max timestep for more visible environment motion", () => {
        expect(environmentScene.timing?.mlsMpmMaxSimulationTimestepS).toBeGreaterThan(1 / 1024);
    });
});
