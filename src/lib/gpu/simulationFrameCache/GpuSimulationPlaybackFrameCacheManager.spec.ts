import { describe, expect, it } from "vitest";

import managerSrc from "./GpuSimulationPlaybackFrameCacheManager.ts?raw";
import shaderSrc from "./simulationPlaybackFrameCache.wgsl?raw";
import { SIMULATION_PLAYBACK_FRAME_BYTES_PER_PARTICLE } from "./GpuSimulationPlaybackFrameCacheManager";

describe("GpuSimulationPlaybackFrameCacheManager", () => {
    it("keeps the playback cache shader independent from the full simulation prelude", () => {
        expect(managerSrc).not.toContain("attachPrelude");
        expect(shaderSrc).not.toContain("uniforms.");
        expect(shaderSrc).toContain("fn particleMaterialFromFlags");
    });

    it("stores only position and material per cached playback particle", () => {
        expect(SIMULATION_PLAYBACK_FRAME_BYTES_PER_PARTICLE).toBe(16);
        expect(shaderSrc).toContain("struct SimulationPlaybackFrameParticle");
        expect(shaderSrc).toContain("pos: vec3f");
        expect(shaderSrc).toContain("material: u32");
    });

    it("marks restored particles as cached identity-determinant frames", () => {
        expect(shaderSrc).toContain("PARTICLE_FLAG_CACHED_IDENTITY_DETERMINANTS = 4u");
        expect(shaderSrc).toContain("| PARTICLE_FLAG_CACHED_IDENTITY_DETERMINANTS");
    });
});
