import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render } from "vitest-browser-svelte";

import type { SimulationState } from "./SimulationState.svelte";
import SimulationTimeline from "./SimulationTimeline.svelte";
import { TIMELINE_FRAME_COUNT } from "./SimulationTimelineTiming";

const createTimelineState = () => {
    const state = {
        timelineFrame: 0,
        timelineFrameCount: TIMELINE_FRAME_COUNT,
        timelineCachedThroughFrame: 0,
        timelineIsBusy: false,
        timelineIsPlaying: false,
        timelineCanSelectCacheDirectory: false,
        timelineStepDivisor: 30,
        timelineDurationS: 6,
        timelineStorageLabel: "test cache",
        timelineFrameByteLength: 1024,
        timelineStatus: "timeline ready",
        stepTimelineFrame: vi.fn(async () => {}),
        pauseTimeline: vi.fn(),
        playTimeline: vi.fn(async () => {}),
        setTimelineFrame: vi.fn(async (value: number) => {
            state.timelineFrame = value;
        }),
        setTimelineStepDivisor: vi.fn(async (value: number) => {
            state.timelineStepDivisor = value;
        }),
        selectTimelineCacheDirectory: vi.fn(async () => {}),
    };

    return state;
};

describe("SimulationTimeline", () => {
    afterEach(() => {
        cleanup();
    });

    it("does not commit typed timestep digits until text entry is committed", async () => {
        const simulationState = createTimelineState();
        await render(SimulationTimeline, {
            props: {
                simulationState: simulationState as unknown as SimulationState,
            },
        });

        const input = document.querySelector("timeline-step input");
        expect(input).toBeInstanceOf(HTMLInputElement);

        const stepInput = input as HTMLInputElement;
        stepInput.focus();
        stepInput.value = "12";
        stepInput.dispatchEvent(new Event("input", { bubbles: true }));
        await Promise.resolve();

        expect(simulationState.setTimelineStepDivisor).not.toHaveBeenCalled();
        expect(document.activeElement).toBe(stepInput);

        stepInput.dispatchEvent(new Event("change", { bubbles: true }));
        await Promise.resolve();

        expect(simulationState.setTimelineStepDivisor).toHaveBeenCalledTimes(1);
        expect(simulationState.setTimelineStepDivisor).toHaveBeenCalledWith(12);
    });

    it("commits dragged timestep changes once on pointer release", async () => {
        const simulationState = createTimelineState();
        await render(SimulationTimeline, {
            props: {
                simulationState: simulationState as unknown as SimulationState,
            },
        });

        const input = document.querySelector("timeline-step input");
        expect(input).toBeInstanceOf(HTMLInputElement);

        const stepInput = input as HTMLInputElement;
        stepInput.dispatchEvent(new PointerEvent("pointerdown", {
            bubbles: true,
            button: 0,
            clientX: 0,
            clientY: 0,
            pointerId: 1,
        }));
        window.dispatchEvent(new PointerEvent("pointermove", {
            bubbles: true,
            clientX: 24,
            clientY: 0,
            pointerId: 1,
        }));
        window.dispatchEvent(new PointerEvent("pointermove", {
            bubbles: true,
            clientX: 48,
            clientY: 0,
            pointerId: 1,
        }));
        await Promise.resolve();

        expect(simulationState.setTimelineStepDivisor).not.toHaveBeenCalled();

        window.dispatchEvent(new PointerEvent("pointerup", {
            bubbles: true,
            clientX: 48,
            clientY: 0,
            pointerId: 1,
        }));
        await Promise.resolve();

        expect(simulationState.setTimelineStepDivisor).toHaveBeenCalledTimes(1);
    });

    it("commits dragged frame scrubber changes once on pointer release", async () => {
        const simulationState = createTimelineState();
        await render(SimulationTimeline, {
            props: {
                simulationState: simulationState as unknown as SimulationState,
            },
        });

        const input = document.querySelector("timeline-range input");
        expect(input).toBeInstanceOf(HTMLInputElement);

        const frameInput = input as HTMLInputElement;
        frameInput.dispatchEvent(new PointerEvent("pointerdown", {
            bubbles: true,
            button: 0,
            clientX: 0,
            clientY: 0,
            pointerId: 1,
        }));

        frameInput.value = "1";
        frameInput.dispatchEvent(new Event("input", { bubbles: true }));
        frameInput.value = "2";
        frameInput.dispatchEvent(new Event("input", { bubbles: true }));
        await Promise.resolve();

        expect(simulationState.setTimelineFrame).not.toHaveBeenCalled();

        window.dispatchEvent(new PointerEvent("pointerup", {
            bubbles: true,
            clientX: 48,
            clientY: 0,
            pointerId: 1,
        }));
        frameInput.dispatchEvent(new Event("change", { bubbles: true }));
        await Promise.resolve();

        expect(simulationState.setTimelineFrame).toHaveBeenCalledTimes(1);
        expect(simulationState.setTimelineFrame).toHaveBeenCalledWith(2);
    });
});
