import { describe, expect, it } from "vitest";

import mpmOpsSrc from "./mpmOps.wgsl?raw";

describe("mpmOps material damping", () => {
    it("does not cancel the grid gravity impulse during particle integration", () => {
        const dampingStart = mpmOpsSrc.indexOf("fn applyMaterialVelocityDamping");
        const nextFunctionStart = mpmOpsSrc.indexOf("fn calculateGridCoordinate", dampingStart);
        const dampingBody = mpmOpsSrc.slice(dampingStart, nextFunctionStart);

        expect(dampingStart).toBeGreaterThanOrEqual(0);
        expect(nextFunctionStart).toBeGreaterThan(dampingStart);
        expect(dampingBody).toContain("materialVelocityDamping");
        expect(dampingBody).toContain("uniforms.gravityDeltaVelocity");
        expect(dampingBody).toContain("(*particle).vel - gravity_delta");
        expect(dampingBody).toContain("+ gravity_delta");
        expect(mpmOpsSrc).not.toContain("materialGravityCancellation");
    });

    it("scales authored damping to the actual simulation substep", () => {
        expect(mpmOpsSrc).toContain("MATERIAL_DAMPING_REFERENCE_TIMESTEP_S = 1.0 / 60.0");
        expect(mpmOpsSrc).toContain("fn materialVelocityDampingPerReferenceStep");
        expect(mpmOpsSrc).toContain("pow(");
        expect(mpmOpsSrc).toContain("uniforms.simulationTimestep / MATERIAL_DAMPING_REFERENCE_TIMESTEP_S");
    });

    it("keeps spawn material values as direct material ids", () => {
        expect(mpmOpsSrc).toContain("fn particleFlagsFromSpawnPoint");
        expect(mpmOpsSrc).toContain("return u32(clamp(round(encodedMaterial), 0.0, 3.0));");
        expect(mpmOpsSrc).toContain("return materialFlagFromId(material);");
    });

    it("keeps environment soil from becoming nearly suspended by per-substep damping", () => {
        const environmentTimestepS = 1 / 256;
        const gravityMPerS2 = 9.81;
        const soilZRetentionPerReferenceStep = 0.82;
        const oldTerminalFallSpeed = gravityMPerS2
            * environmentTimestepS
            * soilZRetentionPerReferenceStep
            / (1 - soilZRetentionPerReferenceStep);
        const scaledRetention = soilZRetentionPerReferenceStep ** (environmentTimestepS / (1 / 60));
        const fixedTerminalFallSpeed = gravityMPerS2
            * environmentTimestepS
            / (1 - scaledRetention);

        expect(oldTerminalFallSpeed).toBeLessThan(0.25);
        expect(fixedTerminalFallSpeed).toBeGreaterThan(0.75);
    });

    it("keeps generated forest materials damped and frictional at the domain floor", () => {
        expect(mpmOpsSrc).toContain("return vec3f(0.78, 0.78, 0.84)");
        expect(mpmOpsSrc).toContain("return vec3f(0.86, 0.86, 0.9)");
        expect(mpmOpsSrc).toContain("return 0.97");
        expect(mpmOpsSrc).toContain("return 0.86");
    });
});
