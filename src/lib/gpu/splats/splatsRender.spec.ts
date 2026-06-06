import { describe, expect, it } from "vitest";

import splatsRenderSrc from "./splatsRender.vert.wgsl?raw";

describe("splatsRender material radius", () => {
    it("keeps generated environment splats smaller than default snow splats", () => {
        expect(splatsRenderSrc).toContain("DEFAULT_SPLAT_RADIUS_SCALE = 0.88");
        expect(splatsRenderSrc).toContain("SOIL_SPLAT_RADIUS_SCALE = 0.76");
        expect(splatsRenderSrc).toContain("BARK_SPLAT_RADIUS_SCALE = 0.72");
        expect(splatsRenderSrc).toContain("LEAF_SPLAT_RADIUS_SCALE = 0.62");
        expect(splatsRenderSrc).toContain("splatRadiusScaleForMaterial(appearance.w)");
    });
});
