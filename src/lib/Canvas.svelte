<script lang="ts">
import { onDestroy, onMount, tick } from "svelte";
import { requestGpuDeviceAndContext } from "./gpu/requestGpuDeviceAndContext";
import { Camera } from "./Camera.svelte";
import { CameraOrbit } from "./CameraOrbit.svelte";
import Draggable, { type Point } from "./Draggable.svelte";
import { GpuSnowPipelineRunner } from "./gpu/GpuSnowPipelineRunner";
import { loadGltfScene } from "./loadScene";
import type { GpuRenderMethodType } from "./gpu/pipelines/GpuRenderMethod";
import type { ElapsedTime } from "./ElapsedTime.svelte";
import type { ColliderGeometry } from "./gpu/GpuSnowPipelineRunner";


let {
    onStatusChange,
    onErr,
    renderMethodType,
    elapsedTime,
}: {
    onStatusChange: (text: string) => void;
    onErr: (text: string) => void;
    renderMethodType: GpuRenderMethodType;
    elapsedTime: ElapsedTime;
} = $props();

let canvas: HTMLCanvasElement;
let width = $state(300);
let height = $state(150);

let nParticles = $state(50_000);
let gridResolution = $state(192);
let simulationTimestepS = $state(1 / 144);

const updateCanvasSize = async () => {
    width = innerWidth;
    height = innerHeight;
};

let stopSimulation: (() => void) | null = null;
let runner: GpuSnowPipelineRunner | null = $state(null);

const orbit = new CameraOrbit();
const camera = new Camera({
    controlScheme: orbit,
    screenDims: { width: () => width, height: () => height },
});

onMount(async () => {
    const response = await requestGpuDeviceAndContext({
        onStatusChange,
        onErr,
        canvas,
    });
    if (response === null) return;
    const { device, context, format, supportsTimestamp } = response;

    onStatusChange("loading geometry...");
    const { vertices } = await loadGltfScene("/monkey.glb"); // particles

    const { positions } = await loadGltfScene("/test.glb"); // static mesh
    const { indices } = await loadGltfScene("/test.glb"); 

    const collider: ColliderGeometry = {
        positions,
        indices,
    };

    runner = new GpuSnowPipelineRunner({
        device,
        format,
        context,
        nParticles,
        gridResolution,
        simulationTimestepS,
        camera,
        meshVertices: vertices,
        collider: collider,
        getRenderMethodType: () => renderMethodType,
        measurePerf: supportsTimestamp,
    });

    updateCanvasSize();

    runner.scatterParticlesInMeshVolume();

    onStatusChange("initializing particles");

    await device.queue.onSubmittedWorkDone(); // need this to set simulation start time accurately

    onStatusChange("off and racing");

    stopSimulation = runner.loop({
        onAnimationFrameTimeUpdate: (ms) =>
            (elapsedTime.animationFrameTimeNs = BigInt(
                Math.round(ms * 1_000_000),
            )),
        onGpuTimeUpdate: (ns) => (elapsedTime.gpuTimeNs = ns),
    });
});



$effect(() => {
    if (runner !== null) {
        runner.resizeTextures(width, height);
    }
});


onDestroy(() => {
    stopSimulation?.();
});
</script>

<!-- <svelte:window onresize={() => updateCanvasSize()} /> -->

<section
    bind:clientWidth={null, (clientWidth) => (width = clientWidth!)}
    bind:clientHeight={null, (clientHeight) => (height = clientHeight!)}
>
    <Draggable
        onDrag={async ({ movement, button, pointerEvent }) => {
            switch (button) {
                case 0:
                    orbit.turn(movement);
                    break;

                case 1:
                    orbit.pan(movement);
                    break;
            }

            pointerEvent.preventDefault();
            // await rerender();
        }}
    >
        {#snippet dragTarget({ onpointerdown })}
            <canvas
                bind:this={canvas}
                {width}
                {height}
                {onpointerdown}
                onwheel={(event) => {
                    orbit.radius *= 2 ** (event.deltaY * 0.001);
                    event.preventDefault();
                }}
            ></canvas>
        {/snippet}
    </Draggable>
</section>

<style lang="scss">
    section {
        width: 100vw;
        height: 100vh;
    }
</style>
