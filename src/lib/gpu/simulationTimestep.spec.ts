import { describe, expect, it } from "vitest";
import {
    calculateCflLimitedSimulationTimestepS,
    calculateSimulationSubstepTimestepS,
    calculateSimulationSubstepsPerMaxStep,
} from "./simulationTimestep";

const gridCellDim = 10 / 384;

describe("simulation timestep CFL subdivision", () => {
    it("keeps a high max timestep as total simulated time when CFL requires smaller substeps", () => {
        const maxSimulationTimestepS = 1 / 50;
        const cflLimitedSimulationTimestepS = calculateCflLimitedSimulationTimestepS({
            maxSimulationTimestepS,
            minGridCellDim: gridCellDim,
            maxCflSpeed: 4,
            externalAcceleration: 9.81,
        });
        const substepsPerMaxStep = calculateSimulationSubstepsPerMaxStep({
            maxSimulationTimestepS,
            cflLimitedSimulationTimestepS,
        });
        const simulationSubstepTimestepS = calculateSimulationSubstepTimestepS({
            maxSimulationTimestepS,
            substepsPerMaxStep,
        });

        expect(cflLimitedSimulationTimestepS).toBeLessThan(maxSimulationTimestepS);
        expect(substepsPerMaxStep).toBeGreaterThan(1);
        expect(simulationSubstepTimestepS).toBeLessThanOrEqual(cflLimitedSimulationTimestepS);
        expect(simulationSubstepTimestepS * substepsPerMaxStep).toBeCloseTo(maxSimulationTimestepS);
    });

    it("uses one substep when CFL does not lower the max timestep", () => {
        const maxSimulationTimestepS = 1 / 1024;
        const cflLimitedSimulationTimestepS = calculateCflLimitedSimulationTimestepS({
            maxSimulationTimestepS,
            minGridCellDim: gridCellDim,
            maxCflSpeed: 0,
            externalAcceleration: 9.81,
        });
        const substepsPerMaxStep = calculateSimulationSubstepsPerMaxStep({
            maxSimulationTimestepS,
            cflLimitedSimulationTimestepS,
        });

        expect(cflLimitedSimulationTimestepS).toBe(maxSimulationTimestepS);
        expect(substepsPerMaxStep).toBe(1);
    });
});
