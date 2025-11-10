<script lang="ts">
import { onDestroy, onMount, tick } from "svelte";
import { requestGpuDeviceAndContext } from "./gpu/requestGpuDeviceAndContext";
import { Camera } from "./Camera.svelte";
import { CameraOrbit } from "./CameraOrbit.svelte";
import Draggable, {type Point} from "./Draggable.svelte";
import { GpuSnowPipelineRunner } from "./gpu/GpuSnowPipelineRunner";

let {
    onStatusChange,
    onErr,
}: {
    onStatusChange: (text: string) => void,
    onErr: (text: string) => void,
} = $props();



let canvas: HTMLCanvasElement;
let width = $state(300);
let height = $state(150);

let nParticles = $state(2_000);
let simulationTimestep = $state(1 / 144);

const updateCanvasSize = async () => {
    width = innerWidth;
    height = innerHeight;
};

let stopSimulation: (() => void) | null;


const orbit = new CameraOrbit();
const camera = new Camera({controlScheme: orbit, screenDims: {width: () => width, height: () => height}});

onMount(async () => {
    const response = await requestGpuDeviceAndContext({onStatusChange, onErr, canvas});
    if (response === null) return;
    const {device, context, format} = response;
    const runner = new GpuSnowPipelineRunner({device, format, context, nParticles, simulationTimestepS: simulationTimestep, camera});

    updateCanvasSize();

    stopSimulation = runner.loop();
});

onDestroy(() => {
    stopSimulation?.();
});
</script>


<svelte:window onresize={() => updateCanvasSize()} />

<Draggable onDrag={async ({movement}) => {
    orbit.move(movement);
    // await rerender();
}}>
    {#snippet dragTarget({onpointerdown})}
        <canvas
            bind:this={canvas}
            {width}
            {height}
            {onpointerdown}
        ></canvas>
    {/snippet}
</Draggable>
