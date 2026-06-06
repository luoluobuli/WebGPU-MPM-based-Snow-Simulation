import { describe, expect, it } from "vitest";
import {
    CFL_NUMBER,
    MAX_CFL_SUBSTEPS_PER_MAX_STEP,
    MAX_SIMULATION_SUBSTEPS_PER_FRAME,
    canRelaxParticleSpeedSampling,
    calculateCflLimitedSimulationTimestepS,
    calculateSimulationFrameSchedule,
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
            maxCflSpeed: 2,
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

    it("subdivides a 256 Hz max step when measured speed exceeds the max-step CFL speed", () => {
        const maxSimulationTimestepS = 1 / 256;
        const maxStepCflSpeed = CFL_NUMBER * gridCellDim / maxSimulationTimestepS;
        const measuredSpeed = maxStepCflSpeed * 1.01;
        const cflLimitedSimulationTimestepS = calculateCflLimitedSimulationTimestepS({
            maxSimulationTimestepS,
            minGridCellDim: gridCellDim,
            maxCflSpeed: measuredSpeed,
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

        expect(substepsPerMaxStep).toBeGreaterThan(1);
        expect(measuredSpeed * simulationSubstepTimestepS).toBeLessThanOrEqual(
            CFL_NUMBER * gridCellDim,
        );
    });

    it("caps pathological CFL subdivision before it can dominate frame work", () => {
        const maxSimulationTimestepS = 1 / 1024;
        const cflLimitedSimulationTimestepS = calculateCflLimitedSimulationTimestepS({
            maxSimulationTimestepS,
            minGridCellDim: gridCellDim,
            maxCflSpeed: 1_000_000,
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

        expect(substepsPerMaxStep).toBe(MAX_CFL_SUBSTEPS_PER_MAX_STEP);
        expect(simulationSubstepTimestepS).toBeCloseTo(
            maxSimulationTimestepS / MAX_CFL_SUBSTEPS_PER_MAX_STEP,
        );
    });

    it("drops backlog when CFL substeps hit the per-frame work cap", () => {
        const schedule = calculateSimulationFrameSchedule({
            timeToSimulateMs: 1_000 / 60,
            maxSimulationTimestepS: 1 / 1024,
            substepsPerMaxStep: MAX_CFL_SUBSTEPS_PER_MAX_STEP,
            oneSimulationStepPerFrame: false,
        });

        expect(schedule.nSubsteps).toBe(MAX_SIMULATION_SUBSTEPS_PER_FRAME);
        expect(schedule.completedMaxSteps).toBe(
            MAX_SIMULATION_SUBSTEPS_PER_FRAME / MAX_CFL_SUBSTEPS_PER_MAX_STEP,
        );
        expect(schedule.shouldDropSimulationBacklog).toBe(true);
    });

    it("keeps normal frame catch-up when CFL does not add substeps", () => {
        const schedule = calculateSimulationFrameSchedule({
            timeToSimulateMs: 1_000 / 60,
            maxSimulationTimestepS: 1 / 1024,
            substepsPerMaxStep: 1,
            oneSimulationStepPerFrame: false,
        });

        expect(schedule.nSubsteps).toBe(18);
        expect(schedule.completedMaxSteps).toBe(18);
        expect(schedule.shouldDropSimulationBacklog).toBe(false);
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
