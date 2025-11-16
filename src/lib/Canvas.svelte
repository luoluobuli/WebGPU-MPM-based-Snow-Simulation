<script lang="ts">
import { onDestroy, onMount, tick } from "svelte";
import { requestGpuDeviceAndContext } from "./gpu/requestGpuDeviceAndContext";
import { Camera } from "./Camera.svelte";
import { CameraOrbit } from "./CameraOrbit.svelte";
import Draggable, {type Point} from "./Draggable.svelte";
import { GpuSnowPipelineRunner } from "./gpu/GpuSnowPipelineRunner";
import { loadGltfScene } from "./loadScene";
    import { samplePointsInMeshVolume } from "./samplePointsInMesh";
    import type { GpuRenderMethodType } from "./gpu/pipelines/GpuRenderMethod";
    import type { GpuElapsedTime } from "./GpuElapsedTime.svelte";

let {
    onStatusChange,
    onErr,
    renderMethodType,
    gpuElapsedTime,
}: {
    onStatusChange: (text: string) => void,
    onErr: (text: string) => void,
    renderMethodType: GpuRenderMethodType,
    gpuElapsedTime: GpuElapsedTime,
} = $props();



let canvas: HTMLCanvasElement;
let width = $state(300);
let height = $state(150);

let nParticles = $state(250);
let gridResolution = $state(8);
let simulationTimestepS = $state(1 / 144);

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
    const {device, context, format, supportsTimestamp} = response;
    
    onStatusChange("loading geometry...");
    const {vertices} = await loadGltfScene("/monkey.glb");
    
    const initialPositions = samplePointsInMeshVolume(vertices, nParticles);
    
    const runner = new GpuSnowPipelineRunner({
        device,
        format,
        context,
        nParticles,
        gridResolution,
        simulationTimestepS,
        camera,
        initialPositions,
        getRenderMethodType: () => renderMethodType,
        measurePerf: supportsTimestamp,
    });

    updateCanvasSize();

    onStatusChange("off and racing");

    stopSimulation = runner.loop({
        onGpuElapsedComputeTimeUpdate: gpuElapsedTimeNs => gpuElapsedTime.computePassNs = gpuElapsedTimeNs,
        onGpuElapsedRenderTimeUpdate: gpuElapsedTimeNs => gpuElapsedTime.renderPassNs = gpuElapsedTimeNs,
    });
});

onDestroy(() => {
    stopSimulation?.();
});
</script>


<!-- <svelte:window onresize={() => updateCanvasSize()} /> -->

<section
    bind:clientWidth={null, clientWidth => width = clientWidth!}
    bind:clientHeight={null, clientHeight => height = clientHeight!}
>
    <Draggable onDrag={async ({movement, button, pointerEvent}) => {
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
    }}>
        {#snippet dragTarget({onpointerdown})}
            <canvas
                bind:this={canvas}
                {width}
                {height}
                {onpointerdown}
                onwheel={event => {
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