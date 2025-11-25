<script lang="ts">
import Canvas from "./Canvas.svelte";
    import Separator from "$lib/components/Separator.svelte";
    import { GpuRenderMethodType } from "$lib/gpu/pipelines/GpuRenderMethod";
    import ElapsedTimeDisplay from "$lib/ElapsedTimeDisplay.svelte";
    import { SimulationState } from "./SimulationState.svelte";
    import { onMount } from "svelte";

let status = $state("loading javascript");
let err = $state<string | null>(null);

let renderMethodType = $state(GpuRenderMethodType.Points);

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
                bind:group={renderMethodType}
                value={GpuRenderMethodType.Points}
                id="render-method-type_points"
            />
            <label for="render-method-type_points">Points</label>
        </div>

        <div>
            <input
                type="radio"
                name="render-method-type"
                bind:group={renderMethodType}
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

        <button>Reset</button>
    </overlays-panel>
</main>
ns


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
    width: 16rem;
    padding: 0.5rem;

    color: oklch(1 0 0);
}
</style>