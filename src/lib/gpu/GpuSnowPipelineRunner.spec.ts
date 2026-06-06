import { describe, expect, it } from "vitest";

import runnerSrc from "./GpuSnowPipelineRunner.svelte.ts?raw";

describe("GpuSnowPipelineRunner timestep stability", () => {
    it("feeds elastic wave speed into CFL instead of a hardcoded stress timestep", () => {
        expect(runnerSrc).toContain("MPM_MAX_ELASTIC_WAVE_SPEED");
        expect(runnerSrc).toContain("elasticWaveSpeed: MPM_MAX_ELASTIC_WAVE_SPEED");
        expect(runnerSrc).not.toContain("STRESS_STABLE_TIMESTEP");
        expect(runnerSrc).not.toContain("stressStableTimestepS");
    });
});
