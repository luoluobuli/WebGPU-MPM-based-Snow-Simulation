<script lang="ts">
import { onMount, tick } from "svelte";
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
// let simulate: (() => Promise<void>) | null = null;
// let render: (() => Promise<void>) | null = null;


// const rerender = async () => {
//     if (render === null) return;

//     onStatusChange("rendering");
//     await render();
//     onStatusChange("done!");
// };

const updateCanvasSizeAndRerender = async () => {
    width = innerWidth;
    height = innerHeight;

    // await tick();
    // await rerender();
};


const orbit = new CameraOrbit();
const camera = new Camera({controlScheme: orbit, screenDims: {width: () => width, height: () => height}});

onMount(async () => {
    const response = await requestGpuDeviceAndContext({onStatusChange, onErr, canvas});
    if (response === null) return;
    const {device, context, format} = response;
    const runner = new GpuSnowPipelineRunner({device, format, context, nParticles, camera});
    const render = () => runner.render();
    const simulate = () => runner.doSimulationStep();

    updateCanvasSizeAndRerender();

    const loop = async () => {
        await simulate();
        await render();

        requestAnimationFrame(loop);
    };

    requestAnimationFrame(loop);
});


</script>


<svelte:window onresize={() => updateCanvasSizeAndRerender()} />

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
