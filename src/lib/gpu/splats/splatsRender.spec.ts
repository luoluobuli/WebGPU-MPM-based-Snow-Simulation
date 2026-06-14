import { describe, expect, it } from "vitest";

import splatsRenderSrc from "./splatsRender.vert.wgsl?raw";

describe("splatsRender material radius", () => {
    it("keeps material-specific splat radius scales explicit", () => {
        expect(splatsRenderSrc).toContain("DEFAULT_SPLAT_RADIUS_SCALE = 0.88");
        expect(splatsRenderSrc).toContain("SOIL_SPLAT_RADIUS_SCALE = 0.76");
        expect(splatsRenderSrc).toContain("BARK_SPLAT_RADIUS_SCALE = 0.72");
        expect(splatsRenderSrc).toContain("LEAF_SPLAT_RADIUS_SCALE = 1.55");
        expect(splatsRenderSrc).toContain("splatRadiusScaleForMaterial(appearance.w)");
    });
});
