<script lang="ts">
import Canvas from "./Canvas.svelte";
import Separator from "$lib/components/generic/Separator.svelte";
import { GpuRenderMethodType } from "$lib/gpu/pipelines/GpuRenderMethod";
import ElapsedTimeDisplay from "./ElapsedTimeDisplay.svelte";
import { SimulationState } from "./SimulationState.svelte";
import { onMount } from "svelte";
import Button from "$lib/components/generic/Button.svelte";
import Hotkey from "$lib/components/headless/Hotkey.svelte";

let status = $state("loading javascript");
let err = $state<string | null>(null);

let canvas = $state<HTMLCanvasElement | null>(null);
let canvasPromise = Promise.withResolvers<HTMLCanvasElement>();

const simulationState = SimulationState.loadOntoCanvas({
    canvasPromise: canvasPromise.promise,
    onStatusChange: text => status = text,
    onErr: text => err = text,
});

onMount(() => {
    canvasPromise.resolve(canvas!);
});

</script>

<svelte:window
    onkeydown={event => {
        switch (event.key) {
            case "r":
                simulationState.reset();
                break;
        }
    }}
/>

<main>
    <Canvas
        {simulationState}
        bind:canvas
    />

    <overlays-panel>
        <h3>Status</h3>
        <div>{err !== null ? `error: ${err}` : status}</div>

        <Separator />

        <h3>Render method</h3>

        <div>
            <input
                type="radio"
                name="render-method-type"
                bind:group={simulationState.renderMethodType}
                value={GpuRenderMethodType.Points}
                id="render-method-type_points"
            />
            <label for="render-method-type_points">Points</label>
        </div>

        <div>
            <input
                type="radio"
                name="render-method-type"
                bind:group={simulationState.renderMethodType}
                value={GpuRenderMethodType.Raymarch}
                id="render-method-type_raymarch"
            />
            <label for="render-method-type_raymarch">Raymarch</label>
        </div>

        <Separator />

        <h3>Elapsed time</h3>

        <dl>
            <dt>Time spent in GPU (sample)</dt>
            <dd>
                <ElapsedTimeDisplay
                    ns={simulationState.elapsedTime.gpuTimeNs}
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

        <Separator />

        <h3>Simulation controls</h3>

        <Hotkey
            key="r"
        >
            {#snippet pressTarget({keyHeld})}
                <Button
                    {keyHeld}
                    onclick={() => simulationState.reset()}
                >Reset (R)</Button>
            {/snippet}
        </Hotkey>
    </overlays-panel>
</main>


<style lang="scss">
main {
    width: 100vw;
    height: 100vh;

    display: grid;

    > :global(*) {
        grid-area: 1/1;
    }
}

overlays-panel {
    width: 20rem;
    padding: 0.5rem;

    line-height: 1.25;

    color: oklch(1 0 0);
}

h3 {
    font-size: 1.5rem;
    margin-bottom: 1rem;
}
</style>