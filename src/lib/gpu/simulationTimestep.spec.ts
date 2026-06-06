import { describe, expect, it } from "vitest";
import {
    CFL_NUMBER,
    ELASTIC_CFL_NUMBER,
    MAX_CFL_SUBSTEPS_PER_MAX_STEP,
    MAX_SIMULATION_SUBSTEPS_PER_FRAME,
    MPM_ELASTIC_MATERIALS,
    MPM_MAX_ELASTIC_WAVE_SPEED,
    canRelaxParticleSpeedSampling,
    calculateCflLimitedSimulationTimestepS,
    calculateElasticWaveCflLimitedTimestepS,
    calculateElasticWaveSpeed,
    calculateSpawnSourceMaxElasticWaveSpeed,
    calculateSimulationFrameSchedule,
    calculateSimulationSubstepTimestepS,
    calculateSimulationSubstepsPerMaxStep,
} from "./simulationTimestep";
import stressTensorOpsSrc from "./shaderPrelude/stressTensorOps.wgsl?raw";
import uniformsManagerSrc from "./uniforms/GpuUniformsBufferManager.ts?raw";

const gridCellDim = 10 / 384;

describe("simulation timestep CFL subdivision", () => {
    it("computes elastic wave speed from Lame stiffness and particle density", () => {
        expect(calculateElasticWaveSpeed({
            shearResistance: 3,
            volumetricResistance: 5,
            density: 2,
        })).toBeCloseTo(Math.sqrt((5 + 2 * 3) / 2));

        expect(calculateElasticWaveSpeed({
            shearResistance: 3,
            volumetricResistance: 5,
            density: 0,
        })).toBe(0);
    });

    it("keeps timestep material constants synchronized with shader stiffness and density", () => {
        expect(stressTensorOpsSrc).toContain("YOUNGS_MODULUS_PA = 1.4e5");
        expect(stressTensorOpsSrc).toContain("POISSONS_RATIO = 0.2");
        expect(stressTensorOpsSrc).toContain("SOIL_SHEAR_RESISTANCE_SCALE = 0.1");
        expect(stressTensorOpsSrc).toContain("SOIL_VOLUME_RESISTANCE_SCALE = 0.14");
        expect(stressTensorOpsSrc).toContain("BARK_SHEAR_RESISTANCE_SCALE = 6.0");
        expect(stressTensorOpsSrc).toContain("BARK_VOLUME_RESISTANCE_SCALE = 8.0");
        expect(stressTensorOpsSrc).toContain("LEAF_SHEAR_RESISTANCE_SCALE = 0.14");
        expect(stressTensorOpsSrc).toContain("LEAF_VOLUME_RESISTANCE_SCALE = 0.18");
        expect(uniformsManagerSrc).toContain("INVERSE_PARTICLE_DENSITY = 1 / 400");

        const [snow, soil, bark, leaf] = MPM_ELASTIC_MATERIALS;

        expect(soil.shearResistance / snow.shearResistance).toBeCloseTo(0.1);
        expect(soil.volumetricResistance / snow.volumetricResistance).toBeCloseTo(0.14);
        expect(bark.shearResistance / snow.shearResistance).toBeCloseTo(6.0);
        expect(bark.volumetricResistance / snow.volumetricResistance).toBeCloseTo(8.0);
        expect(leaf.shearResistance / snow.shearResistance).toBeCloseTo(0.14);
        expect(leaf.volumetricResistance / snow.volumetricResistance).toBeCloseTo(0.18);
        expect(calculateElasticWaveSpeed(bark)).toBeGreaterThan(calculateElasticWaveSpeed(snow));
        expect(MPM_MAX_ELASTIC_WAVE_SPEED).toBeCloseTo(calculateElasticWaveSpeed(bark));
    });

    it("limits elastic CFL to materials actually present in the spawn source", () => {
        const [snow, , bark, leaf] = MPM_ELASTIC_MATERIALS;
        const meshWaveSpeed = calculateSpawnSourceMaxElasticWaveSpeed({
            type: "mesh",
            vertices: [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
        });
        const barkWaveSpeed = calculateSpawnSourceMaxElasticWaveSpeed({
            type: "points",
            points: new Float32Array([
                0, 0, 0, 2,
                0, 0, 0, 3,
            ]),
        });

        expect(meshWaveSpeed).toBeCloseTo(calculateElasticWaveSpeed(snow));
        expect(barkWaveSpeed).toBeCloseTo(
            Math.max(
                calculateElasticWaveSpeed(bark),
                calculateElasticWaveSpeed(leaf),
            ),
        );
    });

    it("derives elastic CFL timestep from material wave speed and grid spacing", () => {
        const elasticLimitedTimestepS = calculateElasticWaveCflLimitedTimestepS({
            minGridCellDim: gridCellDim,
            elasticWaveSpeed: calculateElasticWaveSpeed(MPM_ELASTIC_MATERIALS[0]),
        });

        expect(elasticLimitedTimestepS).toBeCloseTo(
            ELASTIC_CFL_NUMBER * gridCellDim / calculateElasticWaveSpeed(MPM_ELASTIC_MATERIALS[0]),
        );
        expect(elasticLimitedTimestepS).toBeGreaterThan(1 / 1024);
        expect(elasticLimitedTimestepS).toBeLessThan(1 / 384);
    });

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

    it("subdivides large authored MLS-MPM timesteps by elastic-wave CFL", () => {
        const maxSimulationTimestepS = 1 / 384;
        const snowWaveSpeed = calculateElasticWaveSpeed(MPM_ELASTIC_MATERIALS[0]);
        const elasticLimitedTimestepS = calculateElasticWaveCflLimitedTimestepS({
            minGridCellDim: gridCellDim,
            elasticWaveSpeed: snowWaveSpeed,
        });
        const cflLimitedSimulationTimestepS = calculateCflLimitedSimulationTimestepS({
            maxSimulationTimestepS,
            minGridCellDim: gridCellDim,
            maxCflSpeed: 0,
            externalAcceleration: 9.81,
            elasticWaveSpeed: snowWaveSpeed,
        });
        const substepsPerMaxStep = calculateSimulationSubstepsPerMaxStep({
            maxSimulationTimestepS,
            cflLimitedSimulationTimestepS,
        });
        const simulationSubstepTimestepS = calculateSimulationSubstepTimestepS({
            maxSimulationTimestepS,
            substepsPerMaxStep,
        });

        expect(cflLimitedSimulationTimestepS).toBeCloseTo(elasticLimitedTimestepS);
        expect(substepsPerMaxStep).toBe(3);
        expect(simulationSubstepTimestepS).toBeLessThanOrEqual(elasticLimitedTimestepS);
        expect(simulationSubstepTimestepS * substepsPerMaxStep).toBeCloseTo(maxSimulationTimestepS);
    });

    it("leaves the authored 1024 Hz MLS-MPM step unsubdivided by elastic-wave CFL", () => {
        const maxSimulationTimestepS = 1 / 1024;
        const snowWaveSpeed = calculateElasticWaveSpeed(MPM_ELASTIC_MATERIALS[0]);
        const elasticLimitedTimestepS = calculateElasticWaveCflLimitedTimestepS({
            minGridCellDim: gridCellDim,
            elasticWaveSpeed: snowWaveSpeed,
        });
        const cflLimitedSimulationTimestepS = calculateCflLimitedSimulationTimestepS({
            maxSimulationTimestepS,
            minGridCellDim: gridCellDim,
            maxCflSpeed: 0,
            externalAcceleration: 9.81,
            elasticWaveSpeed: snowWaveSpeed,
        });
        const substepsPerMaxStep = calculateSimulationSubstepsPerMaxStep({
            maxSimulationTimestepS,
            cflLimitedSimulationTimestepS,
        });

        expect(elasticLimitedTimestepS).toBeGreaterThan(maxSimulationTimestepS);
        expect(cflLimitedSimulationTimestepS).toBe(maxSimulationTimestepS);
        expect(substepsPerMaxStep).toBe(1);
    });

    it("keeps the 128 grid environment step stable with rigid bark stiffness", () => {
        const environmentGridCellDim = 10 / 128;
        const maxSimulationTimestepS = 1 / 1024;
        const cflLimitedSimulationTimestepS = calculateCflLimitedSimulationTimestepS({
            maxSimulationTimestepS,
            minGridCellDim: environmentGridCellDim,
            maxCflSpeed: 0,
            externalAcceleration: 9.81,
            elasticWaveSpeed: calculateElasticWaveSpeed(MPM_ELASTIC_MATERIALS[2]),
        });
        const substepsPerMaxStep = calculateSimulationSubstepsPerMaxStep({
            maxSimulationTimestepS,
            cflLimitedSimulationTimestepS,
        });

        expect(cflLimitedSimulationTimestepS).toBe(maxSimulationTimestepS);
        expect(substepsPerMaxStep).toBe(1);
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
