<script lang="ts">
import { EntrySlider } from "@vaie/hui";
import type { SimulationState } from "./SimulationState.svelte";
import {
    TIMELINE_MAX_STEP_DIVISOR,
    TIMELINE_MIN_STEP_DIVISOR,
} from "./SimulationTimelineTiming";
import {
    formatReciprocalSecondsDivisor,
    parseReciprocalSecondsDivisor,
} from "./reciprocalSeconds";

let {
    simulationState,
}: {
    simulationState: SimulationState,
} = $props();

const formatBytes = (bytes: number | null) => {
    if (bytes === null) return "---";

    return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
};

const formatSeconds = (seconds: number) => {
    if (seconds < 1) return `${(seconds * 1_000).toFixed(1)} ms`;

    return `${seconds.toFixed(2)} s`;
};

const timelineDisabled = $derived(
    simulationState.timelineIsBusy
    || simulationState.timelineNextUncachedFrame <= 0,
);

type TimelineStepEntryProps = {
    text: string,
    el: HTMLElement | null,
    onElChange: (el: HTMLElement | null) => void,
    elProps: {
        value: string,
        type: "text",
        inputmode: "decimal",
        role: "spinbutton",
        disabled: boolean,
        "aria-disabled": boolean,
        "aria-invalid": boolean,
        "aria-valuemax": number | undefined,
        "aria-valuemin": number | undefined,
        "aria-valuenow": number,
        onblur: () => void,
        onchange: () => void,
        onclick: (event: MouseEvent) => void,
        onfocus: () => void,
        oninput: (event: Event) => void,
        onkeydown: (event: KeyboardEvent) => void,
        onpointerdown: (event: PointerEvent) => void,
    },
    outsideHardBounds: boolean,
    outsideSoftBounds: boolean,
    belowSoftMax: boolean,
    belowSoftMin: boolean,
    disabled: boolean,
    editing: boolean,
    dragging: boolean,
    progress: number,
};

let timelineStepTextInputActive = $state(false);
let timelineStepPointerActive = false;
let timelineStepDraftDivisor = $state<number | null>(null);
let suppressNextTimelineStepCommit = false;
let timelineFramePointerActive = false;
let timelineFrameDraft = $state<number | null>(null);
let suppressNextTimelineFrameCommit = false;

const timelineStepValue = $derived(
    timelineStepDraftDivisor ?? simulationState.timelineStepDivisor,
);
const timelineFrameValue = $derived(
    timelineFrameDraft ?? simulationState.timelineFrame,
);

const clampTimelineFrame = (frameIndex: number) => Math.max(
    0,
    Math.min(simulationState.timelineFrameCount - 1, Math.round(frameIndex)),
);

const timelineFrameFromEvent = (event: Event) => {
    const input = event.currentTarget;
    if (!(input instanceof HTMLInputElement)) return null;
    if (!Number.isFinite(input.valueAsNumber)) return null;

    return clampTimelineFrame(input.valueAsNumber);
};

const resetTimelineStepTextInput = () => {
    timelineStepTextInputActive = false;
    timelineStepDraftDivisor = null;
};

const handleTimelineStepValueChange = (value: number) => {
    if (timelineStepTextInputActive || timelineStepPointerActive) {
        timelineStepDraftDivisor = value;
        return;
    }

    void simulationState.setTimelineStepDivisor(value);
};

const commitTimelineStepText = (text: string) => {
    if (suppressNextTimelineStepCommit) {
        suppressNextTimelineStepCommit = false;
        resetTimelineStepTextInput();
        return;
    }

    if (!timelineStepTextInputActive && timelineStepDraftDivisor === null) return;

    const parsed = parseReciprocalSecondsDivisor(text);
    resetTimelineStepTextInput();
    if (parsed === null) return;

    void simulationState.setTimelineStepDivisor(parsed);
};

const handleTimelineStepInput = (
    event: Event,
    oninput: (event: Event) => void,
) => {
    timelineStepTextInputActive = true;
    oninput(event);
};

const commitTimelineStepDraft = () => {
    const draftDivisor = timelineStepDraftDivisor;
    timelineStepPointerActive = false;
    timelineStepDraftDivisor = null;
    if (draftDivisor === null) return;

    void simulationState.setTimelineStepDivisor(draftDivisor);
};

const handleTimelineStepPointerDown = (
    event: PointerEvent,
    onpointerdown: (event: PointerEvent) => void,
) => {
    timelineStepPointerActive = true;
    timelineStepDraftDivisor = null;
    onpointerdown(event);
};

const handleTimelineStepPointerEnd = () => {
    if (!timelineStepPointerActive) return;

    commitTimelineStepDraft();
};

const commitTimelineFrame = (frameIndex: number) => {
    const clampedFrameIndex = clampTimelineFrame(frameIndex);
    if (clampedFrameIndex === simulationState.timelineFrame) return;

    void simulationState.setTimelineFrame(clampedFrameIndex);
};

const handleTimelineFrameInput = (event: Event) => {
    const frameIndex = timelineFrameFromEvent(event);
    if (frameIndex === null) return;

    timelineFrameDraft = frameIndex;
};

const handleTimelineFramePointerDown = (event: PointerEvent) => {
    if (event.button !== 0) return;

    timelineFramePointerActive = true;
    suppressNextTimelineFrameCommit = false;
    timelineFrameDraft = timelineFrameFromEvent(event) ?? simulationState.timelineFrame;
};

const handleTimelineFramePointerEnd = () => {
    if (!timelineFramePointerActive) return;

    timelineFramePointerActive = false;
    const draftFrame = timelineFrameDraft;
    timelineFrameDraft = null;

    suppressNextTimelineFrameCommit = true;
    setTimeout(() => {
        suppressNextTimelineFrameCommit = false;
    }, 0);

    if (draftFrame === null) return;

    commitTimelineFrame(draftFrame);
};

const handleTimelineFrameCommitEvent = (event: Event) => {
    const frameIndex = timelineFrameFromEvent(event);
    if (frameIndex === null) return;

    if (timelineFramePointerActive) {
        timelineFrameDraft = frameIndex;
        return;
    }

    if (suppressNextTimelineFrameCommit) {
        suppressNextTimelineFrameCommit = false;
        return;
    }

    timelineFrameDraft = null;
    commitTimelineFrame(frameIndex);
};

const handleTimelinePointerEnd = () => {
    handleTimelineStepPointerEnd();
    handleTimelineFramePointerEnd();
};

const handleTimelineStepCommitEvent = (
    event: Event,
    oncommit: () => void,
) => {
    oncommit();

    const input = event.currentTarget;
    if (!(input instanceof HTMLInputElement || input instanceof HTMLTextAreaElement)) {
        return;
    }

    commitTimelineStepText(input.value);
};

const handleTimelineStepKeydown = (
    event: KeyboardEvent,
    onkeydown: (event: KeyboardEvent) => void,
) => {
    if (event.key === "Escape") {
        suppressNextTimelineStepCommit = true;
    }

    onkeydown(event);
};
</script>

<svelte:window
    onpointerup={handleTimelinePointerEnd}
    onpointercancel={handleTimelinePointerEnd}
/>

{#snippet timelineStepEntry({
    text,
    el,
    onElChange,
    elProps,
    outsideHardBounds,
    outsideSoftBounds,
    belowSoftMax,
    belowSoftMin,
    disabled,
    editing,
    dragging,
    progress,
}: TimelineStepEntryProps)}
    <entry-slider
        class:disabled
        class:dragging
        class:editing
        class:outside-hard-bounds={outsideHardBounds}
        class:outside-soft-bounds={outsideSoftBounds}
        class:above-soft-max={belowSoftMax}
        class:below-soft-min={belowSoftMin}
        style:--entry-slider-progress={progress}
    >
        <input
            bind:this={() => el, onElChange}
            {...elProps}
            value={text}
            oninput={(event) => handleTimelineStepInput(event, elProps.oninput)}
            onchange={(event) => handleTimelineStepCommitEvent(event, elProps.onchange)}
            onblur={(event) => handleTimelineStepCommitEvent(event, elProps.onblur)}
            onkeydown={(event) => handleTimelineStepKeydown(event, elProps.onkeydown)}
            onpointerdown={(event) => handleTimelineStepPointerDown(event, elProps.onpointerdown)}
        />
    </entry-slider>
{/snippet}

<timeline-panel aria-label="Simulation timeline">
    <timeline-header>
        <h3>Timeline</h3>

        <timeline-readout aria-live="polite">
            <span>Frame {simulationState.timelineFrame}</span>
            <span>Next uncached {simulationState.timelineNextUncachedFrame}</span>
            <span>Step {formatReciprocalSecondsDivisor(timelineStepValue)}</span>
            <span>Span {formatSeconds(simulationState.timelineDurationS)}</span>
            <span>{simulationState.timelineStorageLabel}</span>
            <span>{formatBytes(simulationState.timelineFrameByteLength)} / frame</span>
        </timeline-readout>
    </timeline-header>

    <timeline-controls>
        <button
            type="button"
            disabled={timelineDisabled || simulationState.timelineFrame === 0}
            onclick={() => void simulationState.stepTimelineFrame(-1)}
        >
            Prev
        </button>

        <button
            type="button"
            disabled={!simulationState.timelineIsPlaying && (
                simulationState.timelineIsBusy
                || simulationState.timelineNextUncachedFrame <= 0
            )}
            onclick={() => simulationState.timelineIsPlaying
                ? simulationState.pauseTimeline()
                : void simulationState.playTimeline()}
        >
            {simulationState.timelineIsPlaying ? "Pause" : "Play"}
        </button>

        <button
            type="button"
            disabled={timelineDisabled || simulationState.timelineFrame >= simulationState.timelineFrameCount - 1}
            onclick={() => void simulationState.stepTimelineFrame(1)}
        >
            Next
        </button>

        <button
            type="button"
            disabled={!simulationState.timelineCanSelectCacheDirectory || simulationState.timelineIsBusy}
            title={simulationState.timelineCanSelectCacheDirectory
                ? "Select cache folder"
                : "Folder cache requires directory picker support"}
            onclick={() => void simulationState.selectTimelineCacheDirectory()}
        >
            Folder
        </button>

        <timeline-step>
            <span>Step</span>
            <EntrySlider
                value={timelineStepValue}
                onValueChange={handleTimelineStepValueChange}
                entry={timelineStepEntry}
                exponential
                min={TIMELINE_MIN_STEP_DIVISOR}
                max={TIMELINE_MAX_STEP_DIVISOR}
                softMin={TIMELINE_MIN_STEP_DIVISOR}
                softMax={TIMELINE_MAX_STEP_DIVISOR}
                step={0.1}
                format={formatReciprocalSecondsDivisor}
                parse={parseReciprocalSecondsDivisor}
                disabled={simulationState.timelineIsBusy}
            />
        </timeline-step>

        <timeline-range>
            <input
                aria-label="Timeline frame"
                type="range"
                min="0"
                max={simulationState.timelineFrameCount - 1}
                step="1"
                value={timelineFrameValue}
                disabled={simulationState.timelineIsBusy || simulationState.timelineNextUncachedFrame <= 0}
                oninput={handleTimelineFrameInput}
                onchange={handleTimelineFrameCommitEvent}
                onpointerdown={handleTimelineFramePointerDown}
            />

            <progress
                aria-label="Cached frames"
                max={simulationState.timelineFrameCount}
                value={simulationState.timelineNextUncachedFrame}
            ></progress>
        </timeline-range>
    </timeline-controls>

    <timeline-status aria-live="polite">
        {simulationState.timelineStatus}
    </timeline-status>
</timeline-panel>


<style lang="scss">
timeline-panel {
    width: min(62rem, calc(100vw - 1rem));
    margin: 0.5rem auto;
    padding: 0.625rem 0.75rem;

    display: grid;
    gap: 0.5rem;

    color: oklch(0.98 0.01 220);
    background: oklch(0.12 0.02 230 / 0.82);
    border: 1px solid oklch(0.92 0.01 220 / 0.45);
    border-radius: 0.5rem;
    box-shadow: 0 0.4rem 1.6rem oklch(0 0 0 / 0.3);
    backdrop-filter: blur(0.5rem);
}

timeline-header {
    display: grid;
    grid-template-columns: auto 1fr;
    align-items: baseline;
    gap: 0.75rem;

    h3 {
        font-size: 1rem;
    }
}

timeline-readout {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 0.375rem 0.75rem;

    color: oklch(0.83 0.03 220);
    font-size: 0.78rem;
    font-variant-numeric: tabular-nums;
}

timeline-controls {
    display: grid;
    grid-template-columns: auto auto auto auto auto minmax(12rem, 1fr);
    align-items: center;
    gap: 0.5rem;
}

button {
    min-width: 4.25rem;
    min-height: 1.875rem;
    padding: 0.25rem 0.65rem;

    color: oklch(0.98 0.01 220);
    background: oklch(0.23 0.04 230 / 0.95);
    border: 1px solid oklch(0.78 0.09 210 / 0.58);
    border-radius: 0.375rem;

    font: inherit;

    cursor: pointer;

    &:hover:not(:disabled),
    &:focus-visible:not(:disabled) {
        background: oklch(0.3 0.06 220 / 0.96);
        border-color: oklch(0.84 0.12 200);
        outline: none;
    }

    &:disabled {
        opacity: 0.45;
        cursor: not-allowed;
    }
}

timeline-step {
    display: grid;
    grid-template-columns: auto 7.5rem;
    align-items: center;
    gap: 0.375rem;

    color: oklch(0.84 0.03 220);
    font-size: 0.78rem;
    font-variant-numeric: tabular-nums;

    :global(entry-slider) {
        width: 7.5rem;
    }

    :global(entry-slider > input) {
        width: 100%;
        min-height: 1.875rem;
        padding: 0.25rem 0.375rem;

        appearance: none;

        color: oklch(0.98 0.01 220);
        background:
            linear-gradient(
                90deg,
                oklch(0.72 0.12 215 / 0.5) 0 calc(var(--entry-slider-progress, 0) * 100%),
                oklch(0.16 0.02 230 / 0.92) calc(var(--entry-slider-progress, 0) * 100%) 100%
            );
        border: 1px solid oklch(0.78 0.09 210 / 0.5);
        border-radius: 0.375rem;
        outline: none;

        font: inherit;
        font-variant-numeric: tabular-nums;
        text-align: right;

        cursor: ew-resize;
    }

    :global(entry-slider > input:hover),
    :global(entry-slider:focus-within > input) {
        border-color: oklch(0.84 0.12 200);
        box-shadow: 0 0 0 0.12rem oklch(0.72 0.12 215 / 0.35);
    }

    :global(entry-slider.dragging > input) {
        border-color: oklch(0.78 0.16 155);
        box-shadow: 0 0 0 0.12rem oklch(0.7 0.14 155 / 0.3);
    }

    :global(entry-slider.editing > input) {
        cursor: text;
    }

    :global(entry-slider.disabled > input) {
        opacity: 0.58;
        cursor: not-allowed;
    }
}

timeline-range {
    min-width: 0;

    display: grid;
    gap: 0.25rem;

    input,
    progress {
        width: 100%;
    }

    input {
        accent-color: oklch(0.76 0.15 155);
        cursor: pointer;

        &:disabled {
            cursor: not-allowed;
        }
    }

    progress {
        height: 0.35rem;

        accent-color: oklch(0.72 0.13 155);
    }
}

timeline-status {
    min-height: 1em;

    color: oklch(0.84 0.04 155);
    font-size: 0.78rem;
    font-variant-numeric: tabular-nums;
}

@media (max-width: 48rem) {
    timeline-header,
    timeline-controls {
        grid-template-columns: 1fr;
    }

    timeline-readout {
        justify-content: flex-start;
    }

    timeline-controls {
        grid-template-columns: repeat(4, minmax(0, 1fr));
    }

    timeline-step,
    timeline-range {
        grid-column: 1 / -1;
    }

    button {
        min-width: 0;
    }
}
</style>
