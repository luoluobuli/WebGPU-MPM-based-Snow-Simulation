export const CFL_NUMBER = 0.5;
export const MIN_SIMULATION_TIMESTEP_S = 1e-6;

// CFL subdivision improves accuracy, but the shader displacement clamp is the
// stability fallback when extreme speeds would otherwise create unbounded work.
export const MAX_SIMULATION_DRIFT_MS = 250;
export const MAX_SIMULATION_STEPS_PER_FRAME = 128;
export const MAX_SIMULATION_SUBSTEPS_PER_FRAME = 64;
export const MAX_CFL_SUBSTEPS_PER_MAX_STEP = 4;

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
