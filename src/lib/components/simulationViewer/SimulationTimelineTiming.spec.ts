import { describe, expect, it } from "vitest";

import {
    TIMELINE_DEFAULT_SECONDS_PER_FRAME,
    TIMELINE_FRAME_INTERVAL_MS,
    calculateTimelineBakeSubsteps,
    calculateTimelinePlaybackDelayMs,
    calculateTimelinePlaybackFrame,
    timelineSecondsPerFrameFromStepDivisor,
    timelineStepDivisorFromSecondsPerFrame,
    timelineFrameSimulationTimeS,
} from "./SimulationTimelineTiming";

describe("Simulation timeline timing", () => {
    it("maps timeline frames to simulated seconds instead of one solver step", () => {
        expect(calculateTimelineBakeSubsteps({
            currentSimulationTimeS: 0,
            targetFrame: 30,
            simulationTimestepS: 1 / 1024,
        })).toBe(1024);
    });

    it("maps timeline frames with a configurable simulated step size", () => {
        expect(timelineFrameSimulationTimeS(10, 0.1)).toBe(1);
        expect(timelineSecondsPerFrameFromStepDivisor(10)).toBe(0.1);
        expect(timelineStepDivisorFromSecondsPerFrame(0.1)).toBe(10);
        expect(calculateTimelineBakeSubsteps({
            currentSimulationTimeS: 0,
            targetFrame: 10,
            simulationTimestepS: 0.01,
            secondsPerFrame: 0.1,
        })).toBe(100);
        expect(timelineFrameSimulationTimeS(1)).toBe(TIMELINE_DEFAULT_SECONDS_PER_FRAME);
    });

    it("chooses playback frames from elapsed wall-clock time", () => {
        expect(calculateTimelinePlaybackFrame({
            startFrame: 0,
            elapsedMs: TIMELINE_FRAME_INTERVAL_MS * 30,
            frameCount: 180,
        })).toBe(30);
    });

    it("waits until the next frame is due instead of adding a fixed post-restore delay", () => {
        expect(calculateTimelinePlaybackDelayMs({
            startFrame: 0,
            currentFrame: 0,
            playbackStartMs: 100,
            nowMs: 100,
        })).toBeCloseTo(TIMELINE_FRAME_INTERVAL_MS);

        expect(calculateTimelinePlaybackDelayMs({
            startFrame: 0,
            currentFrame: 10,
            playbackStartMs: 100,
            nowMs: 100 + TIMELINE_FRAME_INTERVAL_MS * 20,
        })).toBe(0);
    });
});
