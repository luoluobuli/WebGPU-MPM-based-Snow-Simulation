import { describe, expect, it } from "vitest";

import particleScatterSrc from "./particleScatter.cs.wgsl?raw";

describe("particleScatter anchored spawn materials", () => {
    it("decodes support flags from procedural spawn-point material values", () => {
        expect(particleScatterSrc).toContain("let spawnPointMaterial = spawnPoint.w");
        expect(particleScatterSrc).toContain("particleMaterialFromSpawnPoint(spawnPointMaterial)");
        expect(particleScatterSrc).toContain("particleFlagsFromSpawnPoint(spawnPointMaterial)");
    });
});
