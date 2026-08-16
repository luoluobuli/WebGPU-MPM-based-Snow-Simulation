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
    kinematicFinalScene = null,
}: {
    scene?: SimulationSceneConfig,
    kinematicScene?: SimulationSceneConfig | null,
    kinematicFinalScene?: SimulationSceneConfig | null,
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
const final = kinematic && urlSearchParams?.has("final") === true;
const noColliderControl =
    final && urlSearchParams?.has("control") === true;
const inspect = kinematic && urlSearchParams?.has("inspect") === true;
const inspectContact = inspect
    && urlSearchParams?.get("inspect") === "contact";
const getActiveScene = (): SimulationSceneConfig => {
    const inspectCollider = inspect
        && urlSearchParams?.has("collider") === true;
    const activeScene = final && kinematicFinalScene !== null
        ? kinematicFinalScene
        : kinematic && kinematicScene !== null
            ? kinematicScene
            : scene;
    const simulationScene = noColliderControl
        ? {
            ...activeScene,
            colliderSource: null,
            renderCollider: false,
        }
        : activeScene;
    if (!inspect) return simulationScene;

    return {
        ...simulationScene,
        renderCollider: inspectCollider,
        camera: inspectContact
            ? {
                radius: 0.65,
                lat: Math.PI * 0.49,
                long: Math.PI * 0.5,
                offset: [0.28, 1.34, 0.04],
            }
            : {
                radius: 2.7,
                lat: Math.PI * 0.4,
                long: Math.PI * 0.5,
                offset: [0, 0.9, 0.05],
            },
    };
};

const simulationState = SimulationState.loadOntoCanvas({
    getScene: getActiveScene,
    getPairedTimelineScene: () =>
        final ? kinematicFinalScene : null,
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
