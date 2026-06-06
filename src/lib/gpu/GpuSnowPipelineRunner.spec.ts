import { describe, expect, it } from "vitest";

import runnerSrc from "./GpuSnowPipelineRunner.svelte.ts?raw";

describe("GpuSnowPipelineRunner timestep stability", () => {
    it("feeds scene material elastic wave speed into CFL instead of a hardcoded stress timestep", () => {
        expect(runnerSrc).toContain("calculateSpawnSourceMaxElasticWaveSpeed");
        expect(runnerSrc).toContain("this.elasticWaveSpeed = calculateSpawnSourceMaxElasticWaveSpeed(spawnSource)");
        expect(runnerSrc).toContain("elasticWaveSpeed: this.elasticWaveSpeed");
        expect(runnerSrc).not.toContain("STRESS_STABLE_TIMESTEP");
        expect(runnerSrc).not.toContain("stressStableTimestepS");
    });
});
