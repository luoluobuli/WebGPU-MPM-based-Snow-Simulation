<script lang="ts">
import Canvas from "./Canvas.svelte";
import { SimulationState } from "./SimulationState.svelte";
import { onMount } from "svelte";
import SimulationStatusPanel from "./SimulationStatusPanel.svelte";
import SimulationControlPanel from "./SimulationControlPanel.svelte";

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

<main>
    <Canvas
        {simulationState}
        bind:canvas
    />

    <simulation-overlay-panels>
        <SimulationControlPanel
            {simulationState}
        />

        <SimulationStatusPanel
            {simulationState}
            {status}
            {err}
        />
    </simulation-overlay-panels>
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

simulation-overlay-panels {
    display: grid;
    grid-template-columns: auto 1fr auto;

    pointer-events: none;

    > :global(*) {
        pointer-events: auto;
    }

    > :global(:nth-child(1)) {
        grid-area: 1/1;
    }

    > :global(:nth-child(2)) {
        grid-area: 1/3;
    }
}
</style>