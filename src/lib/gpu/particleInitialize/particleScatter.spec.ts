import { describe, expect, it } from "vitest";

import particleScatterSrc from "./particleScatter.cs.wgsl?raw";

describe("particleScatter spawn materials", () => {
    it("decodes particle material flags from spawn-point material values", () => {
        expect(particleScatterSrc).toContain("let spawnPointMaterial = spawnPoint.w");
        expect(particleScatterSrc).toContain("particleMaterialFromSpawnPoint(spawnPointMaterial)");
        expect(particleScatterSrc).toContain("particleFlagsFromSpawnPoint(spawnPointMaterial)");
    });
});
