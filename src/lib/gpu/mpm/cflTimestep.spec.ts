import { describe, expect, it } from "vitest";

import fusedMlsSrc from "./fusedMlsG2p2g.cs.wgsl?raw";
import gridToParticleSrc from "./gridToParticle.cs.wgsl?raw";
import gridUpdateSrc from "./gridUpdate.cs.wgsl?raw";
import integrateParticlesSrc from "./integrateParticles.wgsl?raw";
import mpmOpsSrc from "../shaderPrelude/mpmOps.wgsl?raw";
import particleToGridSrc from "./particleToGrid.cs.wgsl?raw";

describe("MPM CFL velocity handling", () => {
    it("does not enforce CFL by clipping grid or transfer velocities", () => {
        const fusedWithoutIntegration = fusedMlsSrc.replace(
            /fn integrateFusedParticle[\s\S]*?fn scatterMlsParticleToGrid/,
            "fn scatterMlsParticleToGrid",
        );
        const shaderSources = [
            fusedWithoutIntegration,
            gridToParticleSrc,
            gridUpdateSrc,
            mpmOpsSrc,
            particleToGridSrc,
        ].join("\n");

        expect(shaderSources).not.toContain("limitVelocityToCfl");
        expect(shaderSources).not.toContain("maxStableParticleSpeed()");
        expect(shaderSources).not.toContain("maxStableParticleSpeedSquared()");
        expect(shaderSources).not.toContain("uniforms.maxStableParticleSpeed");
        expect(shaderSources).not.toContain("uniforms.maxStableParticleSpeedSquared");
    });

    it("records pre-clamp speeds when integration applies a CFL displacement safety clamp", () => {
        expect(integrateParticlesSrc).toContain("let unclamped_speed_squared");
        expect(integrateParticlesSrc).toContain("uniforms.maxStableParticleSpeed");
        expect(integrateParticlesSrc).toContain("speed_squared = unclamped_speed_squared");
        expect(fusedMlsSrc).toContain("let unclamped_speed_squared");
        expect(fusedMlsSrc).toContain("uniforms.maxStableParticleSpeed");
        expect(fusedMlsSrc).toContain("speed_squared = unclamped_speed_squared");
    });

    it("keeps only fixed-point overflow guards in transfer velocity clamps", () => {
        expect(mpmOpsSrc).toContain("MAX_FIXED_POINT_GRID_SPEED");
        expect(gridUpdateSrc).toContain("clampVelocityToFixedPointRange");
        expect(gridUpdateSrc).toContain("maxFixedPointGridSpeed()");
        expect(gridToParticleSrc).toContain("maxFixedPointGridSpeed()");
        expect(particleToGridSrc).toContain("particle_mass_fixed_units * maxFixedPointGridSpeed()");
        expect(fusedMlsSrc).toContain("maxFixedPointGridSpeed()");
    });
});
