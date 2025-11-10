<script lang="ts">
import { onMount, tick } from "svelte";
import { requestGpuDeviceAndContext } from "./gpu/requestGpuDeviceAndContext";
import { setupGpuPipelines } from "./gpu/setupGpuPipelines";
import { createGpuRenderer } from "./gpu/createGpuRenderer";
    import { Camera } from "./Camera.svelte";
    import { CameraOrbit } from "./CameraOrbit.svelte";

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
let render: () => Promise<void>;


const rerender = async () => {
    onStatusChange("rendering");
    await render();
    onStatusChange("done!");
};

const updateCanvasSizeAndRerender = async () => {
    width = innerWidth;
    height = innerHeight;

    await tick();
    await rerender();
};


const orbit = new CameraOrbit();
const camera = new Camera({controlScheme: orbit, screenDims: {width: () => width, height: () => height}});

onMount(async () => {
    const response = await requestGpuDeviceAndContext({onStatusChange, onErr, canvas});
    if (response === null) return;

    const {device, context, format} = response;
    const {particlePosBuffer, uniformsBuffer, renderBindGroup, renderPipeline} = setupGpuPipelines({device, format, nParticles});
    render = createGpuRenderer({device, context, nParticles, renderBindGroup, renderPipeline, particlePosBuffer, uniformsBuffer, camera});

    updateCanvasSizeAndRerender();
});
</script>


<svelte:window onresize={() => updateCanvasSizeAndRerender()} />

<canvas
    bind:this={canvas}
    {width}
    {height}
></canvas>