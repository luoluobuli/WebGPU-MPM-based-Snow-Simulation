export const TIMELINE_FRAME_COUNT = 180;
export const TIMELINE_FRAME_RATE = 30;
export const TIMELINE_FRAME_INTERVAL_MS = 1_000 / TIMELINE_FRAME_RATE;
export const TIMELINE_DEFAULT_SECONDS_PER_FRAME = 1 / TIMELINE_FRAME_RATE;
export const TIMELINE_MIN_SECONDS_PER_FRAME = 1 / 240;
export const TIMELINE_MAX_SECONDS_PER_FRAME = 1;
export const TIMELINE_MIN_STEP_DIVISOR = 1 / TIMELINE_MAX_SECONDS_PER_FRAME;
export const TIMELINE_MAX_STEP_DIVISOR = 1 / TIMELINE_MIN_SECONDS_PER_FRAME;

export const clampTimelineSecondsPerFrame = (secondsPerFrame: number) => {
    if (!Number.isFinite(secondsPerFrame)) return TIMELINE_DEFAULT_SECONDS_PER_FRAME;

    return Math.max(
        TIMELINE_MIN_SECONDS_PER_FRAME,
        Math.min(TIMELINE_MAX_SECONDS_PER_FRAME, secondsPerFrame),
    );
};

export const timelineStepDivisorFromSecondsPerFrame = (secondsPerFrame: number) =>
    1 / clampTimelineSecondsPerFrame(secondsPerFrame);

export const timelineSecondsPerFrameFromStepDivisor = (stepDivisor: number) => {
    if (!Number.isFinite(stepDivisor) || stepDivisor <= 0) {
        return TIMELINE_MAX_SECONDS_PER_FRAME;
    }

    return clampTimelineSecondsPerFrame(1 / stepDivisor);
};

export const timelineFrameSimulationTimeS = (
    frameIndex: number,
    secondsPerFrame = TIMELINE_DEFAULT_SECONDS_PER_FRAME,
) =>
    Math.max(
        0,
        Math.round(frameIndex) * clampTimelineSecondsPerFrame(secondsPerFrame),
    );

export const calculateTimelineBakeSubsteps = ({
    currentSimulationTimeS,
    targetFrame,
    simulationTimestepS,
    secondsPerFrame = TIMELINE_DEFAULT_SECONDS_PER_FRAME,
}: {
    currentSimulationTimeS: number,
    targetFrame: number,
    simulationTimestepS: number,
    secondsPerFrame?: number,
}) => {
    if (
        !Number.isFinite(currentSimulationTimeS)
        || !Number.isFinite(targetFrame)
        || !Number.isFinite(simulationTimestepS)
        || simulationTimestepS <= 0
    ) {
        return 0;
    }

    const targetSimulationTimeS = timelineFrameSimulationTimeS(
        targetFrame,
        secondsPerFrame,
    );
    const remainingSimulationTimeS = targetSimulationTimeS - currentSimulationTimeS;
    if (remainingSimulationTimeS <= 0) return 0;

    return Math.max(1, Math.round(remainingSimulationTimeS / simulationTimestepS));
};

export const calculateTimelinePlaybackFrame = ({
    startFrame,
    elapsedMs,
    frameCount = TIMELINE_FRAME_COUNT,
    frameIntervalMs = TIMELINE_FRAME_INTERVAL_MS,
}: {
    startFrame: number,
    elapsedMs: number,
    frameCount?: number,
    frameIntervalMs?: number,
}) => {
    if (
        !Number.isFinite(startFrame)
        || !Number.isFinite(elapsedMs)
        || !Number.isFinite(frameCount)
        || !Number.isFinite(frameIntervalMs)
        || frameCount <= 0
        || frameIntervalMs <= 0
    ) {
        return 0;
    }

    const elapsedFrames = Math.floor(Math.max(0, elapsedMs) / frameIntervalMs);
    const maxFrame = Math.max(0, Math.floor(frameCount) - 1);

    return Math.max(
        0,
        Math.min(
            maxFrame,
            Math.round(startFrame) + elapsedFrames,
        ),
    );
};

export const calculateTimelinePlaybackDelayMs = ({
    startFrame,
    currentFrame,
    playbackStartMs,
    nowMs,
    frameIntervalMs = TIMELINE_FRAME_INTERVAL_MS,
}: {
    startFrame: number,
    currentFrame: number,
    playbackStartMs: number,
    nowMs: number,
    frameIntervalMs?: number,
}) => {
    if (
        !Number.isFinite(startFrame)
        || !Number.isFinite(currentFrame)
        || !Number.isFinite(playbackStartMs)
        || !Number.isFinite(nowMs)
        || !Number.isFinite(frameIntervalMs)
        || frameIntervalMs <= 0
    ) {
        return 0;
    }

    const displayedFrames = Math.max(0, Math.round(currentFrame) - Math.round(startFrame) + 1);
    const nextFrameTimeMs = playbackStartMs + displayedFrames * frameIntervalMs;

    return Math.max(0, nextFrameTimeMs - nowMs);
};
