import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render } from "vitest-browser-svelte";

import type { SimulationState } from "./SimulationState.svelte";
import SimulationTimeline from "./SimulationTimeline.svelte";
import { TIMELINE_FRAME_COUNT } from "./SimulationTimelineTiming";

const createTimelineState = ({
    timelineNextUncachedFrame = 1,
    timelineIsBusy = false,
    timelineIsPlaying = false,
}: {
    timelineNextUncachedFrame?: number,
    timelineIsBusy?: boolean,
    timelineIsPlaying?: boolean,
} = {}) => {
    const state = {
        timelineFrame: 0,
        timelineFrameCount: TIMELINE_FRAME_COUNT,
        timelineNextUncachedFrame,
        timelineIsBusy,
        timelineIsPlaying,
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

    it("shows empty cache progress without negative frame numbers", async () => {
        const simulationState = createTimelineState({
            timelineNextUncachedFrame: 0,
        });
        await render(SimulationTimeline, {
            props: {
                simulationState: simulationState as unknown as SimulationState,
            },
        });

        const text = document.body.textContent ?? "";
        expect(text).toContain("Next uncached 0");
        expect(text).not.toContain("-1");

        const frameInput = document.querySelector("timeline-range input");
        expect(frameInput).toBeInstanceOf(HTMLInputElement);
        expect((frameInput as HTMLInputElement).disabled).toBe(true);

        const progress = document.querySelector("timeline-range progress");
        expect(progress).toBeInstanceOf(HTMLProgressElement);
        expect((progress as HTMLProgressElement).max).toBe(TIMELINE_FRAME_COUNT);
        expect((progress as HTMLProgressElement).value).toBe(0);
    });

    it("keeps the frame scrubber enabled while playback is changing frames", async () => {
        const simulationState = createTimelineState({
            timelineIsBusy: true,
            timelineIsPlaying: true,
        });
        await render(SimulationTimeline, {
            props: {
                simulationState: simulationState as unknown as SimulationState,
            },
        });

        const frameInput = document.querySelector("timeline-range input");
        expect(frameInput).toBeInstanceOf(HTMLInputElement);
        expect((frameInput as HTMLInputElement).disabled).toBe(false);
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

    it("requests dragged frame scrubber changes before pointer release", async () => {
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

        expect(simulationState.setTimelineFrame).toHaveBeenCalledTimes(2);
        expect(simulationState.setTimelineFrame).toHaveBeenNthCalledWith(1, 1);
        expect(simulationState.setTimelineFrame).toHaveBeenNthCalledWith(2, 2);

        window.dispatchEvent(new PointerEvent("pointerup", {
            bubbles: true,
            clientX: 48,
            clientY: 0,
            pointerId: 1,
        }));
        frameInput.dispatchEvent(new Event("change", { bubbles: true }));
        await Promise.resolve();

        expect(simulationState.setTimelineFrame).toHaveBeenCalledTimes(2);
    });

    it("keeps showing the committed scrubber frame while the seek loads", async () => {
        let finishSeek = () => {};
        const seekPromise = new Promise<void>((resolve) => {
            finishSeek = resolve;
        });
        const simulationState = createTimelineState();
        simulationState.setTimelineFrame = vi.fn(async () => {
            await seekPromise;
        });

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

        frameInput.value = "2";
        frameInput.dispatchEvent(new Event("input", { bubbles: true }));
        await Promise.resolve();

        window.dispatchEvent(new PointerEvent("pointerup", {
            bubbles: true,
            clientX: 48,
            clientY: 0,
            pointerId: 1,
        }));
        frameInput.dispatchEvent(new Event("change", { bubbles: true }));
        await Promise.resolve();

        expect(simulationState.timelineFrame).toBe(0);
        expect(frameInput.value).toBe("2");

        finishSeek();
    });
});
