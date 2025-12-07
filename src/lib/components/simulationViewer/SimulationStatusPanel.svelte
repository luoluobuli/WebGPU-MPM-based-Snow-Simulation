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
    <dl>
        <dt>GPU simulation compute pass (sample)</dt>
        <dd>
            <ElapsedTimeDisplay
                ns={simulationState.elapsedTime.gpuComputeSimulationStepTimeNs}
            />
        </dd>
    </dl>

    <h4>Render</h4>

    <div>{nPrerenderPasses ?? "(unknown)"} prerender {nPrerenderPasses === 1 ? "pass" : "passes"}</div>

    <dl>
        {#each simulationState.prerenderElapsedTimes as prerenderElapsedTime}
            <dt>GPU {prerenderElapsedTime.label} pass (sample)</dt>
            <dd>
                <ElapsedTimeDisplay
                    ns={prerenderElapsedTime.elapsedTimeNs ?? 0n}
                />
            </dd>
        {/each}

        <dt>GPU render pass (sample)</dt>
        <dd>
            <ElapsedTimeDisplay
                ns={simulationState.elapsedTime.gpuRenderTimeNs}
            />
        </dd>
    </dl>

    <h4>Overall</h4>
    <dl>
        <dt>Total animation frame time</dt>
        <dd>
            <ElapsedTimeDisplay
                ns={simulationState.elapsedTime.animationFrameTimeNs}
                showMsFractionalPart={false}
                inverseLabel="fps"
            />
        </dd>
    </dl>

</OverlayPanel>