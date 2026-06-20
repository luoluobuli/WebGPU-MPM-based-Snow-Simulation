<script lang="ts">
import Canvas from "./Canvas.svelte";
import { SimulationState } from "./SimulationState.svelte";
import { onMount } from "svelte";
import SimulationStatusPanel from "./SimulationStatusPanel.svelte";
import SimulationControlPanel from "./SimulationControlPanel.svelte";
import SimulationTimeline from "./SimulationTimeline.svelte";
import {
    defaultSimulationScene,
    type SimulationSceneConfig,
} from "./SimulationScene";
import { browser } from "$app/environment";

let {
    scene = defaultSimulationScene,
    kinematicScene = null,
}: {
    scene?: SimulationSceneConfig,
    kinematicScene?: SimulationSceneConfig | null,
} = $props();

let status = $state("loading javascript");
let err = $state<string | null>(null);

let canvas = $state<HTMLCanvasElement | null>(null);
let canvasPromise = Promise.withResolvers<HTMLCanvasElement>();

const urlSearchParams = browser
    ? new URLSearchParams(location.search)
    : null;
const timeline = urlSearchParams?.has("timeline") === true;
const kinematic = timeline && urlSearchParams?.has("kinematic") === true;
const getActiveScene = () => kinematic && kinematicScene !== null
    ? kinematicScene
    : scene;

const simulationState = SimulationState.loadOntoCanvas({
    getScene: getActiveScene,
    timeline,
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

        {#if timeline}
            <SimulationTimeline
                {simulationState}
            />
        {/if}
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
    width: 100vw;
    height: 100vh;

    display: grid;
    grid-template-columns: auto 1fr auto;
    grid-template-rows: minmax(0, 1fr) auto;
    overflow: hidden;

    pointer-events: none;

    > :global(*) {
        pointer-events: auto;
    }

    > :global(:nth-child(1)) {
        grid-area: 1/1;
        align-self: stretch;
        min-height: 0;
        max-height: 100%;
    }

    > :global(:nth-child(2)) {
        grid-area: 1/3;
        align-self: stretch;
        min-height: 0;
        max-height: 100%;
    }

    > :global(:nth-child(3)) {
        grid-area: 2/1/3/4;
        align-self: end;
        justify-self: center;
    }
}
</style>
