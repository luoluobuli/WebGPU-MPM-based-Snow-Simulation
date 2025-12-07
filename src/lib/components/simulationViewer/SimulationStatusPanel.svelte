<script lang="ts">
import ElapsedTimeDisplay from "./ElapsedTimeDisplay.svelte";
import OverlayPanel from "./OverlayPanel.svelte";
import Separator from "$lib/components/generic/Separator.svelte";
import type { SimulationState } from "./SimulationState.svelte";

let {
    simulationState,
    status,
    err,
}: {
    simulationState: SimulationState,
    status: string,
    err: string | null,
} = $props();

const nPrerenderPasses = $derived(simulationState.prerenderElapsedTimes?.length);
</script>

<OverlayPanel>
    <h3>Status</h3>
    <div>{err !== null ? `error: ${err}` : status}</div>

    <Separator />

    <h3>Performance</h3>

    <h4>Simulation</h4>

    <perf-item>
        <perf-label>GPU simulation compute pass (sample)</perf-label>
        <ElapsedTimeDisplay
            ns={simulationState.elapsedTime.gpuComputeSimulationStepTimeNs}
        />
    </perf-item>

    <h4>Render</h4>

    <div>{nPrerenderPasses ?? "(unknown)"} prerender {nPrerenderPasses === 1 ? "pass" : "passes"}</div>

    {#each simulationState.prerenderElapsedTimes as prerenderElapsedTime (prerenderElapsedTime.label)}
    <perf-item>
        <perf-label>GPU {prerenderElapsedTime.label} pass (sample)</perf-label>
        <ElapsedTimeDisplay
            ns={prerenderElapsedTime.elapsedTimeNs ?? 0n}
        />
    </perf-item>
    {/each}

    <perf-item>
        <perf-label>GPU composite render pass (sample)</perf-label>
        <ElapsedTimeDisplay
            ns={simulationState.elapsedTime.gpuRenderTimeNs}
        />
    </perf-item>

    <h4>Overall</h4>
    <perf-item>
        <perf-label>Total animation frame time</perf-label>
        <ElapsedTimeDisplay
            ns={simulationState.elapsedTime.animationFrameTimeNs}
            showMsFractionalPart={false}
            inverseLabel="fps"
        />
    </perf-item>
</OverlayPanel>