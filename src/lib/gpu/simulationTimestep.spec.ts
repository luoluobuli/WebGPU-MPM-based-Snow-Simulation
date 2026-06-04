import { describe, expect, it } from "vitest";
import {
    canRelaxParticleSpeedSampling,
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

    it("relaxes speed sampling when acceleration cannot consume CFL headroom", () => {
        expect(canRelaxParticleSpeedSampling({
            maxSimulationTimestepS: 1 / 1024,
            minGridCellDim: gridCellDim,
            latestMaxParticleSpeed: 2,
            externalAcceleration: 9.81,
            relaxedSampleIntervalFrames: 24,
            speedHeadroom: 0.8,
            oneSimulationStepPerFrame: true,
        })).toBe(true);
    });

    it("keeps frequent speed sampling near the CFL speed threshold", () => {
        expect(canRelaxParticleSpeedSampling({
            maxSimulationTimestepS: 1 / 192,
            minGridCellDim: gridCellDim,
            latestMaxParticleSpeed: 1,
            externalAcceleration: 9.81,
            relaxedSampleIntervalFrames: 24,
            speedHeadroom: 0.8,
            oneSimulationStepPerFrame: true,
        })).toBe(false);
    });

    it("keeps frequent speed sampling when backlog frames can advance multiple max steps", () => {
        expect(canRelaxParticleSpeedSampling({
            maxSimulationTimestepS: 1 / 1024,
            minGridCellDim: gridCellDim,
            latestMaxParticleSpeed: 0,
            externalAcceleration: 9.81,
            relaxedSampleIntervalFrames: 24,
            speedHeadroom: 0.8,
            oneSimulationStepPerFrame: false,
        })).toBe(false);
    });
});
