export const CFL_NUMBER = 0.5;
export const MIN_SIMULATION_TIMESTEP_S = 1e-6;

export const calculateCflLimitedSimulationTimestepS = ({
    maxSimulationTimestepS,
    minGridCellDim,
    maxCflSpeed,
    externalAcceleration,
}: {
    maxSimulationTimestepS: number,
    minGridCellDim: number,
    maxCflSpeed: number,
    externalAcceleration: number,
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

    return Math.max(
        MIN_SIMULATION_TIMESTEP_S,
        Math.min(
            safeMaxSimulationTimestepS,
            speedLimitedTimestepS,
            accelerationLimitedTimestepS,
        ),
    );
};

export const calculateSimulationSubstepsPerMaxStep = ({
    maxSimulationTimestepS,
    cflLimitedSimulationTimestepS,
}: {
    maxSimulationTimestepS: number,
    cflLimitedSimulationTimestepS: number,
}) => {
    if (
        !Number.isFinite(maxSimulationTimestepS)
        || !Number.isFinite(cflLimitedSimulationTimestepS)
        || maxSimulationTimestepS <= 0
        || cflLimitedSimulationTimestepS <= 0
    ) {
        return 1;
    }

    return Math.max(1, Math.ceil(maxSimulationTimestepS / cflLimitedSimulationTimestepS));
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
