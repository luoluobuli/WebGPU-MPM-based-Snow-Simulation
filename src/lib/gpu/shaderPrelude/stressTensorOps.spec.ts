import { describe, expect, it } from "vitest";

import stressTensorOpsSrc from "./stressTensorOps.wgsl?raw";

describe("stressTensorOps forest material stiffness", () => {
    it("keeps generated bark and leaves stiffer than loose soil", () => {
        expect(stressTensorOpsSrc).toContain("SOIL_SHEAR_RESISTANCE_SCALE = 0.1");
        expect(stressTensorOpsSrc).toContain("BARK_SHEAR_RESISTANCE_SCALE = 0.5");
        expect(stressTensorOpsSrc).toContain("LEAF_SHEAR_RESISTANCE_SCALE = 0.14");
        expect(stressTensorOpsSrc).toContain("BARK_VOLUME_RESISTANCE_SCALE = 0.62");
        expect(stressTensorOpsSrc).toContain("LEAF_VOLUME_RESISTANCE_SCALE = 0.18");
    });
});
