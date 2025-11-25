<script lang="ts">
import Canvas from "./Canvas.svelte";
import { SimulationState } from "./SimulationState.svelte";
import { onMount } from "svelte";
import SimulationStatusPanel from "./SimulationStatusPanel.svelte";

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

    <SimulationStatusPanel
        {simulationState}
        {status}
        {err}
    />
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
</style>