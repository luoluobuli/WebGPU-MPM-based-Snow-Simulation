import { describe, expect, it } from "vitest";

import clearGridSource from "./clearGridAccumulator.wgsl?raw";
import fusedSource from "./fusedMlsG2p2g.cs.wgsl?raw";
import pipelineSource from "./GpuMpmPipelineManager.ts?raw";
import particleToGridSource from "./particleToGrid.cs.wgsl?raw";

describe("fused MLS FLIP/PIC transfer", () => {
    it("keeps a paired stress-free transfer grid through bootstrap and ping-pong", () => {
        expect(pipelineSource).toContain("nextGridVelocityBuffer");
        expect(pipelineSource).toContain(
            "WRITE_FLIP_TRANSFER: 1",
        );
        expect(particleToGridSource).toContain(
            "grid_transfer_accumulator",
        );
        expect(clearGridSource).toContain(
            "grid_transfer_accumulator",
        );
        expect(
            clearGridSource.indexOf("grid_transfer_accumulator[cell_index]"),
        ).toBeLessThan(clearGridSource.lastIndexOf("}"));

        expect(fusedSource).toContain(
            "next_grid_transfer_accumulator",
        );
    });

    it("excludes stress from the old grid velocity and includes it in the new velocity", () => {
        expect(particleToGridSource).toContain(
            "let transfer_affine_fixed_units = "
            + "particle_mass_fixed_units * affine_velocity",
        );
        expect(particleToGridSource).toContain(
            "mls_affine_fixed_units = mls_affine_fixed_units "
            + "+ mls_stress_affine_fixed_units",
        );
        expect(fusedSource).toContain(
            "post_force_velocity - pre_force_velocity",
        );
    });

    it("uses Disney's 95 percent FLIP and 5 percent PIC blend", () => {
        expect(fusedSource).toContain(
            "const FUSED_MLS_FLIP_RATIO = 0.95",
        );
        expect(fusedSource).toContain(
            "particle_velocity + weighted_grid_velocity_delta",
        );
        expect(fusedSource).toContain(
            "(1.0 - FUSED_MLS_FLIP_RATIO) * pic_velocity",
        );
    });

    it("retains occupied zero-velocity cells in the gather", () => {
        expect(fusedSource).toContain("tileCellHasMass");
        expect(fusedSource).toContain(
            "select(0.0, 1.0, cell_velocity.has_mass)",
        );
    });
});
