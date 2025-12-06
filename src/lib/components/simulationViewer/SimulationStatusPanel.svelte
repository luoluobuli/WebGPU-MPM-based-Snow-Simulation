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
</script>

<OverlayPanel>
    <h3>Status</h3>
    <div>{err !== null ? `error: ${err}` : status}</div>

    <Separator />

    <h3>Elapsed time</h3>

    <dl>
        <dt>GPU simulation compute pass (sample)</dt>
        <dd>
            <ElapsedTimeDisplay
                ns={simulationState.elapsedTime.gpuComputeSimulationStepTimeNs}
                inverseLabel="commands / s"
            />
        </dd>

        <dt>GPU prerender compute pass (sample)</dt>
        <dd>
            <ElapsedTimeDisplay
                ns={simulationState.elapsedTime.gpuComputePrerenderTimeNs}
                inverseLabel="commands / s"
            />
        </dd>

        <dt>GPU render pass (sample)</dt>
        <dd>
            <ElapsedTimeDisplay
                ns={simulationState.elapsedTime.gpuRenderTimeNs}
                inverseLabel="commands / s"
            />
        </dd>

        <dt>Total animation frame time</dt>
        <dd>
            <ElapsedTimeDisplay
                ns={simulationState.elapsedTime.animationFrameTimeNs}
                showMsFractionalPart={false}
            />
        </dd>
    </dl>

</OverlayPanel>