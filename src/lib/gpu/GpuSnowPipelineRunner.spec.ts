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

    it("renders cached playback frames in one restore-and-render submission", () => {
        const methodStart = runnerSrc.indexOf("renderSimulationPlaybackFrame(");
        const methodEnd = runnerSrc.indexOf("renderStillFrame({");
        const methodSrc = runnerSrc.slice(methodStart, methodEnd);

        expect(methodStart).toBeGreaterThanOrEqual(0);
        expect(methodEnd).toBeGreaterThan(methodStart);
        expect(methodSrc).toContain("measureGpuTimestamps = false");
        expect(methodSrc).toContain("this.simulationPlaybackFrameCacheManager.addRestoreDispatch({ commandEncoder });");
        expect(methodSrc).toContain("this.addRender(commandEncoder, shouldMeasureGpuTimestamps);");
        expect(methodSrc.match(/this\.device\.queue\.submit/g)).toHaveLength(1);
    });
});
