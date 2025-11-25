import { onDestroy, onMount } from "svelte";
import { GpuSnowPipelineRunner } from "./gpu/GpuSnowPipelineRunner";
import { requestGpuDeviceAndContext } from "./gpu/requestGpuDeviceAndContext";
import { loadGltfScene } from "./loadScene";
import modelUrl from "$lib/assets/models/monkey.glb?url";
import { CameraOrbit } from "./CameraOrbit.svelte";
import { Camera } from "./Camera.svelte";
import { ElapsedTime } from "./ElapsedTime.svelte";
import { GpuRenderMethodType } from "./gpu/pipelines/GpuRenderMethod";

export class SimulationState {
    width = $state(300);
    height = $state(150);

    nParticles = $state(500_000);
    gridResolution = $state(192);
    simulationTimestepS = $state(1 / 144);

    renderMethodType = $state(GpuRenderMethodType.Points);

    readonly orbit = new CameraOrbit();
    readonly camera = new Camera({
        controlScheme: this.orbit,
        screenDims: { width: () => this.width, height: () => this.height },
    });

    readonly elapsedTime = new ElapsedTime();


    static loadOntoCanvas({
        canvasPromise,
        onStatusChange,
        onErr,
    }: {
        canvasPromise: Promise<HTMLCanvasElement>,
        onStatusChange: (status: string) => void,
        onErr: (err: string) => void,
    }) {
        const state = new SimulationState();


        let stopSimulation: (() => void) | null = null;
        let runner: GpuSnowPipelineRunner | null = $state(null);

        onMount(async () => {
            const response = await requestGpuDeviceAndContext({
                onStatusChange,
                onErr,
                canvas: await canvasPromise,
            });
            if (response === null) return;
            const { device, context, format, supportsTimestamp } = response;

            onStatusChange("loading geometry...");
            const { vertices } = await loadGltfScene(modelUrl);

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


            state.width = innerWidth;
            state.height = innerHeight;


            runner = new GpuSnowPipelineRunner({
                device,
                format,
                context,
                nParticles: state.nParticles,
                gridResolution: state.gridResolution,
                simulationTimestepS: state.simulationTimestepS,
                camera: state.camera,
                meshVertices: vertices,
                colliderVertices: colliderVertices,
                colliderIndices: colliderIndices,
                getRenderMethodType: () => state.renderMethodType,
                measurePerf: supportsTimestamp,
            });

            runner.scatterParticlesInMeshVolume();

            onStatusChange("initializing particles");

            await device.queue.onSubmittedWorkDone(); // need this to set simulation start time accurately

            onStatusChange("off and racing");

            stopSimulation = runner.loop({
                onAnimationFrameTimeUpdate: (ms) =>
                    (state.elapsedTime.animationFrameTimeNs = BigInt(
                        Math.round(ms * 1_000_000),
                    )),
                onGpuTimeUpdate: (ns) => (state.elapsedTime.gpuTimeNs = ns),
            });


            return state;
        });

        onDestroy(() => {
            stopSimulation?.();
        });


        $effect(() => {
            runner?.resizeTextures(state.width, state.height);
        });


        return state;
    }
}