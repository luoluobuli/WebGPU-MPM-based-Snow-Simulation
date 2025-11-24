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

let nParticles = $state(500_000);
let gridResolution = $state(192);
let simulationTimestepS = $state(1 / 144);

const updateCanvasSize = async () => {
    width = innerWidth;
    height = innerHeight;
};

let stopSimulation: (() => void) | null;

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
    const { vertices } = await loadGltfScene("/monkey.glb");

    const colliderVertices = new Float32Array([
        1, 0, 1,   // 0
        2, 0, 1,   // 1
        2, 1, 1,   // 2
        1, 1, 1,   // 3

        1, 0, 2,   // 4
        2, 0, 2,   // 5
        2, 1, 2,   // 6
        1, 1, 2    // 7
    ]);

    const colliderIndices = new Uint32Array([
        // Front
        0, 1, 2,
        0, 2, 3,

        // Back
        5, 4, 7,
        5, 7, 6,

        // Left
        4, 0, 3,
        4, 3, 7,

        // Right
        1, 5, 6,
        1, 6, 2,

        // Top
        3, 2, 6,
        3, 6, 7,

        // Bottom
        4, 5, 1,
        4, 1, 0
    ]);

    const runner = new GpuSnowPipelineRunner({
        device,
        format,
        context,
        nParticles,
        gridResolution,
        simulationTimestepS,
        camera,
        meshVertices: vertices,
        colliderVertices: colliderVertices,
        colliderIndices: colliderIndices,
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
