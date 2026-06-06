import type { SpawnPointSource } from "./particleInitialize/GpuSpawnVolumeBufferManager";

export const CFL_NUMBER = 0.5;
export const ELASTIC_CFL_NUMBER = 0.75;
export const MIN_SIMULATION_TIMESTEP_S = 1e-6;

// CFL subdivision improves accuracy, but the shader displacement clamp is the
// stability fallback when extreme speeds would otherwise create unbounded work.
export const MAX_SIMULATION_DRIFT_MS = 250;
export const MAX_SIMULATION_STEPS_PER_FRAME = 128;
export const MAX_SIMULATION_SUBSTEPS_PER_FRAME = 64;
export const MAX_CFL_SUBSTEPS_PER_MAX_STEP = 128;

export type ElasticMaterial = {
    shearResistance: number,
    volumetricResistance: number,
    density: number,
};

const MPM_YOUNGS_MODULUS_PA = 1.4e5;
const MPM_POISSONS_RATIO = 0.2;
const MPM_SHEAR_RESISTANCE = MPM_YOUNGS_MODULUS_PA / (2 * (1 + MPM_POISSONS_RATIO));
const MPM_VOLUME_RESISTANCE = MPM_YOUNGS_MODULUS_PA
    * MPM_POISSONS_RATIO
    / ((1 + MPM_POISSONS_RATIO) * (1 - 2 * MPM_POISSONS_RATIO));
const MPM_PARTICLE_DENSITY = 400;
const MPM_PARTICLE_MATERIAL_DEFAULT = 0;
const MPM_PARTICLE_MATERIAL_SOIL = 1;
const MPM_PARTICLE_MATERIAL_BARK = 2;
const MPM_PARTICLE_MATERIAL_LEAF = 3;
const MPM_SOIL_SHEAR_RESISTANCE_SCALE = 0.1;
const MPM_SOIL_VOLUME_RESISTANCE_SCALE = 0.14;
const MPM_BARK_SHEAR_RESISTANCE_SCALE = 6.0;
const MPM_BARK_VOLUME_RESISTANCE_SCALE = 8.0;
const MPM_LEAF_SHEAR_RESISTANCE_SCALE = 0.14;
const MPM_LEAF_VOLUME_RESISTANCE_SCALE = 0.18;

export const MPM_ELASTIC_MATERIALS: ElasticMaterial[] = [
    {
        shearResistance: MPM_SHEAR_RESISTANCE,
        volumetricResistance: MPM_VOLUME_RESISTANCE,
        density: MPM_PARTICLE_DENSITY,
    },
    {
        shearResistance: MPM_SHEAR_RESISTANCE * MPM_SOIL_SHEAR_RESISTANCE_SCALE,
        volumetricResistance: MPM_VOLUME_RESISTANCE * MPM_SOIL_VOLUME_RESISTANCE_SCALE,
        density: MPM_PARTICLE_DENSITY,
    },
    {
        shearResistance: MPM_SHEAR_RESISTANCE * MPM_BARK_SHEAR_RESISTANCE_SCALE,
        volumetricResistance: MPM_VOLUME_RESISTANCE * MPM_BARK_VOLUME_RESISTANCE_SCALE,
        density: MPM_PARTICLE_DENSITY,
    },
    {
        shearResistance: MPM_SHEAR_RESISTANCE * MPM_LEAF_SHEAR_RESISTANCE_SCALE,
        volumetricResistance: MPM_VOLUME_RESISTANCE * MPM_LEAF_VOLUME_RESISTANCE_SCALE,
        density: MPM_PARTICLE_DENSITY,
    },
];

export const calculateElasticWaveSpeed = ({
    shearResistance,
    volumetricResistance,
    density,
}: ElasticMaterial) => {
    if (
        !Number.isFinite(shearResistance)
        || !Number.isFinite(volumetricResistance)
        || !Number.isFinite(density)
        || shearResistance < 0
        || volumetricResistance < 0
        || density <= 0
    ) {
        return 0;
    }

    return Math.sqrt((volumetricResistance + 2 * shearResistance) / density);
};

export const calculateMaxElasticWaveSpeed = (materials: ElasticMaterial[]) => {
    return materials.reduce(
        (maxSpeed, material) => Math.max(maxSpeed, calculateElasticWaveSpeed(material)),
        0,
    );
};

export const MPM_MAX_ELASTIC_WAVE_SPEED = calculateMaxElasticWaveSpeed(MPM_ELASTIC_MATERIALS);

export const particleMaterialFromSpawnValue = (encodedMaterial: number) => {
    if (!Number.isFinite(encodedMaterial)) {
        return MPM_PARTICLE_MATERIAL_DEFAULT;
    }

    return Math.max(
        MPM_PARTICLE_MATERIAL_DEFAULT,
        Math.min(
            MPM_PARTICLE_MATERIAL_LEAF,
            Math.round(encodedMaterial),
        ),
    );
};

export const calculateMaxElasticWaveSpeedForMaterialIds = (materialIds: Iterable<number>) => {
    let maxSpeed = 0;
    let sawMaterial = false;

    for (const materialId of materialIds) {
        const material = MPM_ELASTIC_MATERIALS[
            particleMaterialFromSpawnValue(materialId)
        ];
        if (material === undefined) {
            continue;
        }

        sawMaterial = true;
        maxSpeed = Math.max(maxSpeed, calculateElasticWaveSpeed(material));
    }

    return sawMaterial
        ? maxSpeed
        : calculateElasticWaveSpeed(MPM_ELASTIC_MATERIALS[MPM_PARTICLE_MATERIAL_DEFAULT]);
};

export const calculateSpawnSourceMaxElasticWaveSpeed = (source: SpawnPointSource) => {
    if (source.type === "mesh") {
        return calculateMaxElasticWaveSpeedForMaterialIds([
            MPM_PARTICLE_MATERIAL_DEFAULT,
        ]);
    }

    const materialIds = new Set<number>();
    for (let i = 3; i < source.points.length; i += 4) {
        materialIds.add(particleMaterialFromSpawnValue(source.points[i]));
    }

    return calculateMaxElasticWaveSpeedForMaterialIds(materialIds);
};

export const calculateElasticWaveCflLimitedTimestepS = ({
    minGridCellDim,
    elasticWaveSpeed,
    elasticCflNumber = ELASTIC_CFL_NUMBER,
}: {
    minGridCellDim: number,
    elasticWaveSpeed: number,
    elasticCflNumber?: number,
}) => {
    if (
        !Number.isFinite(minGridCellDim)
        || !Number.isFinite(elasticWaveSpeed)
        || !Number.isFinite(elasticCflNumber)
        || minGridCellDim <= 0
        || elasticWaveSpeed <= 0
        || elasticCflNumber <= 0
    ) {
        return Number.POSITIVE_INFINITY;
    }

    return elasticCflNumber * minGridCellDim / elasticWaveSpeed;
};

export const calculateCflLimitedSimulationTimestepS = ({
    maxSimulationTimestepS,
    minGridCellDim,
    maxCflSpeed,
    externalAcceleration,
    elasticWaveSpeed = 0,
    elasticCflNumber = ELASTIC_CFL_NUMBER,
}: {
    maxSimulationTimestepS: number,
    minGridCellDim: number,
    maxCflSpeed: number,
    externalAcceleration: number,
    elasticWaveSpeed?: number,
    elasticCflNumber?: number,
}) => {
    const safeMaxSimulationTimestepS = Number.isFinite(maxSimulationTimestepS)
        ? Math.max(maxSimulationTimestepS, MIN_SIMULATION_TIMESTEP_S)
        : MIN_SIMULATION_TIMESTEP_S;

    const speedLimitedTimestepS = maxCflSpeed > 0
        ? CFL_NUMBER * minGridCellDim / maxCflSpeed
        : Number.POSITIVE_INFINITY;

    const accelerationLimitedTimestepS = externalAcceleration > 0
        ? Math.sqrt(CFL_NUMBER * minGridCellDim / externalAcceleration)
        : Number.POSITIVE_INFINITY;

    const elasticLimitedTimestepS = calculateElasticWaveCflLimitedTimestepS({
        minGridCellDim,
        elasticWaveSpeed,
        elasticCflNumber,
    });

    return Math.max(
        MIN_SIMULATION_TIMESTEP_S,
        Math.min(
            safeMaxSimulationTimestepS,
            speedLimitedTimestepS,
            accelerationLimitedTimestepS,
            elasticLimitedTimestepS,
        ),
    );
};

export const calculateSimulationSubstepsPerMaxStep = ({
    maxSimulationTimestepS,
    cflLimitedSimulationTimestepS,
    maxSubstepsPerMaxStep = MAX_CFL_SUBSTEPS_PER_MAX_STEP,
}: {
    maxSimulationTimestepS: number,
    cflLimitedSimulationTimestepS: number,
    maxSubstepsPerMaxStep?: number,
}) => {
    if (
        !Number.isFinite(maxSimulationTimestepS)
        || !Number.isFinite(cflLimitedSimulationTimestepS)
        || !Number.isFinite(maxSubstepsPerMaxStep)
        || maxSimulationTimestepS <= 0
        || cflLimitedSimulationTimestepS <= 0
        || maxSubstepsPerMaxStep <= 0
    ) {
        return 1;
    }

    return Math.min(
        Math.max(1, Math.floor(maxSubstepsPerMaxStep)),
        Math.max(1, Math.ceil(maxSimulationTimestepS / cflLimitedSimulationTimestepS)),
    );
};

export const calculateSimulationSubstepTimestepS = ({
    maxSimulationTimestepS,
    substepsPerMaxStep,
}: {
    maxSimulationTimestepS: number,
    substepsPerMaxStep: number,
}) => {
    if (
        !Number.isFinite(maxSimulationTimestepS)
        || maxSimulationTimestepS <= 0
        || !Number.isFinite(substepsPerMaxStep)
        || substepsPerMaxStep <= 0
    ) {
        return MIN_SIMULATION_TIMESTEP_S;
    }

    return Math.max(
        MIN_SIMULATION_TIMESTEP_S,
        maxSimulationTimestepS / substepsPerMaxStep,
    );
};

export const canRelaxParticleSpeedSampling = ({
    maxSimulationTimestepS,
    minGridCellDim,
    latestMaxParticleSpeed,
    externalAcceleration,
    relaxedSampleIntervalFrames,
    speedHeadroom,
    oneSimulationStepPerFrame,
}: {
    maxSimulationTimestepS: number,
    minGridCellDim: number,
    latestMaxParticleSpeed: number,
    externalAcceleration: number,
    relaxedSampleIntervalFrames: number,
    speedHeadroom: number,
    oneSimulationStepPerFrame: boolean,
}) => {
    if (
        !oneSimulationStepPerFrame
        || !Number.isFinite(maxSimulationTimestepS)
        || !Number.isFinite(minGridCellDim)
        || !Number.isFinite(latestMaxParticleSpeed)
        || !Number.isFinite(externalAcceleration)
        || !Number.isFinite(relaxedSampleIntervalFrames)
        || !Number.isFinite(speedHeadroom)
        || maxSimulationTimestepS <= 0
        || minGridCellDim <= 0
        || relaxedSampleIntervalFrames <= 0
        || speedHeadroom <= 0
    ) {
        return false;
    }

    const speedLimitedMaxStepThreshold = CFL_NUMBER * minGridCellDim / maxSimulationTimestepS;
    const maxUnobservedSpeedGrowth =
        Math.max(0, externalAcceleration)
        * maxSimulationTimestepS
        * relaxedSampleIntervalFrames;

    return Math.max(0, latestMaxParticleSpeed) + maxUnobservedSpeedGrowth
        < speedLimitedMaxStepThreshold * Math.min(1, speedHeadroom);
};

export const calculateSimulationFrameSchedule = ({
    timeToSimulateMs,
    maxSimulationTimestepS,
    substepsPerMaxStep,
    oneSimulationStepPerFrame,
    maxSimulationDriftMs = MAX_SIMULATION_DRIFT_MS,
    maxSimulationStepsPerFrame = MAX_SIMULATION_STEPS_PER_FRAME,
    maxSimulationSubstepsPerFrame = MAX_SIMULATION_SUBSTEPS_PER_FRAME,
}: {
    timeToSimulateMs: number,
    maxSimulationTimestepS: number,
    substepsPerMaxStep: number,
    oneSimulationStepPerFrame: boolean,
    maxSimulationDriftMs?: number,
    maxSimulationStepsPerFrame?: number,
    maxSimulationSubstepsPerFrame?: number,
}) => {
    const safeSubstepsPerMaxStep = Number.isFinite(substepsPerMaxStep)
        ? Math.max(1, Math.floor(substepsPerMaxStep))
        : 1;
    const safeMaxSimulationStepsPerFrame = Number.isFinite(maxSimulationStepsPerFrame)
        ? Math.max(0, Math.floor(maxSimulationStepsPerFrame))
        : 0;
    const safeMaxSimulationSubstepsPerFrame = Number.isFinite(maxSimulationSubstepsPerFrame)
        ? Math.max(0, Math.floor(maxSimulationSubstepsPerFrame))
        : 0;

    if (
        !Number.isFinite(timeToSimulateMs)
        || !Number.isFinite(maxSimulationTimestepS)
        || !Number.isFinite(maxSimulationDriftMs)
        || maxSimulationTimestepS <= 0
        || maxSimulationDriftMs < 0
    ) {
        return {
            nSteps: 0,
            nSubsteps: 0,
            completedMaxSteps: 0,
            shouldDropSimulationBacklog: true,
        };
    }

    const safeTimeToSimulateMs = Math.max(0, timeToSimulateMs);
    const maxSimulationTimestepMs = maxSimulationTimestepS * 1_000;
    let nSteps = 0;
    let shouldDropSimulationBacklog = false;

    if (oneSimulationStepPerFrame) {
        nSteps = safeTimeToSimulateMs > 0 ? 1 : 0;
        shouldDropSimulationBacklog = true;
    }
    else if (safeTimeToSimulateMs <= maxSimulationDriftMs) {
        nSteps = Math.min(
            Math.ceil(safeTimeToSimulateMs / maxSimulationTimestepMs),
            safeMaxSimulationStepsPerFrame,
        );
    }
    else {
        shouldDropSimulationBacklog = true;
    }

    const requestedSubsteps = nSteps * safeSubstepsPerMaxStep;
    const nSubsteps = Math.min(requestedSubsteps, safeMaxSimulationSubstepsPerFrame);
    const completedMaxSteps = nSubsteps / safeSubstepsPerMaxStep;

    return {
        nSteps,
        nSubsteps,
        completedMaxSteps,
        shouldDropSimulationBacklog:
            shouldDropSimulationBacklog
            || nSubsteps < requestedSubsteps,
    };
};
