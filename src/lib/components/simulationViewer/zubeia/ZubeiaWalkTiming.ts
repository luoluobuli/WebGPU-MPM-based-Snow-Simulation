export const ZUBEIA_WALK_ANIMATION_NAME = "walk_forward";

type ZubeiaWalkTimingInput = {
    timeS: number,
    cycleDurationS: number,
    durationS: number,
    stridePerCycle: readonly [number, number, number],
};

export type ZubeiaWalkTiming = {
    gaitTimeS: number,
    rootTranslation: [number, number, number],
};

const assertFinitePositive = (value: number, label: string) => {
    if (!Number.isFinite(value) || value <= 0) {
        throw new Error(`${label} must be finite and positive`);
    }
};

export const sampleZubeiaWalkTiming = ({
    timeS,
    cycleDurationS,
    durationS,
    stridePerCycle,
}: ZubeiaWalkTimingInput): ZubeiaWalkTiming => {
    if (!Number.isFinite(timeS)) {
        throw new Error("timeS must be finite");
    }

    assertFinitePositive(cycleDurationS, "cycleDurationS");
    assertFinitePositive(durationS, "durationS");

    const clampedTimeS = Math.min(Math.max(timeS, 0), durationS);
    const isFinalPose = clampedTimeS === durationS;
    const gaitTimeS = isFinalPose
        ? cycleDurationS
        : clampedTimeS % cycleDurationS;
    const cycleProgress = clampedTimeS / cycleDurationS;

    return {
        gaitTimeS,
        rootTranslation: [
            stridePerCycle[0] * cycleProgress,
            stridePerCycle[1] * cycleProgress,
            stridePerCycle[2] * cycleProgress,
        ],
    };
};
