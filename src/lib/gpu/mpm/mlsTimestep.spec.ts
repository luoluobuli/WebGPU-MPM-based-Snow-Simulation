import { describe, expect, it } from "vitest";

import fusedMlsSrc from "./fusedMlsG2p2g.cs.wgsl?raw";
import gridToParticleSrc from "./gridToParticle.cs.wgsl?raw";
import integrateParticlesSrc from "./integrateParticles.wgsl?raw";
import particleToGridSrc from "./particleToGrid.cs.wgsl?raw";
import uniformsManagerSrc from "../uniforms/GpuUniformsBufferManager.ts?raw";

describe("MLS-MPM timestep invariance", () => {
    it("stores velocity gradients instead of timestep-scaled deformation deltas", () => {
        expect(gridToParticleSrc).toContain("sanitizeVelocityGradient");
        expect(fusedMlsSrc).toContain("sanitizeVelocityGradient");
        expect(integrateParticlesSrc).toContain("deformationDeltaFromVelocityGradient");
        expect(fusedMlsSrc).toContain("deformationDeltaFromVelocityGradient");
    });

    it("does not reconstruct MLS affine velocity by dividing by the current timestep", () => {
        expect(particleToGridSrc).not.toContain("deformation_displacement * uniforms.invSimulationTimestep");
        expect(fusedMlsSrc).not.toContain("deformation_displacement * uniforms.invSimulationTimestep");
    });

    it("keeps G2P velocity-gradient scales independent of timestep", () => {
        expect(uniformsManagerSrc).toContain("upload[4] = 4 * this.invGridCellDims[0]");
        expect(uniformsManagerSrc).toContain("upload[5] = 4 * this.invGridCellDims[1]");
        expect(uniformsManagerSrc).toContain("upload[6] = 4 * this.invGridCellDims[2]");
        expect(uniformsManagerSrc).toContain("upload[7] = 1");
    });
});
