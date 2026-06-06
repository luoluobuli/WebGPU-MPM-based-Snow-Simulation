import { describe, expect, it } from "vitest";

import stressTensorOpsSrc from "./stressTensorOps.wgsl?raw";

describe("stressTensorOps forest material stiffness", () => {
    it("keeps generated bark rigid while leaves stay softer than trunks", () => {
        expect(stressTensorOpsSrc).toContain("SOIL_SHEAR_RESISTANCE_SCALE = 0.1");
        expect(stressTensorOpsSrc).toContain("BARK_SHEAR_RESISTANCE_SCALE = 6.0");
        expect(stressTensorOpsSrc).toContain("LEAF_SHEAR_RESISTANCE_SCALE = 0.14");
        expect(stressTensorOpsSrc).toContain("BARK_VOLUME_RESISTANCE_SCALE = 8.0");
        expect(stressTensorOpsSrc).toContain("LEAF_VOLUME_RESISTANCE_SCALE = 0.18");
    });
});
