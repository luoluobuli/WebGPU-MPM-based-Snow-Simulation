import { onDestroy, onMount } from "svelte";
import { GpuSnowPipelineRunner } from "../../gpu/GpuSnowPipelineRunner";
import { requestGpuDeviceAndContext } from "../../gpu/requestGpuDeviceAndContext";
import { loadGltfScene } from "./loadScene";
import modelUrl from "$lib/assets/models/monkey.glb?url";
import colliderUrl from "$lib/assets/models/test.glb?url";
import { CameraOrbit } from "./CameraOrbit.svelte";
import { Camera } from "./Camera.svelte";
import { ElapsedTime } from "./ElapsedTime.svelte";
import { GpuRenderMethodType } from "$lib/gpu/GpuRenderMethod";
import type { ColliderGeometry } from "../../gpu/GpuSnowPipelineRunner";

export class SimulationState {
    width = $state(300);
    height = $state(150);

    nParticles = $state(500_000);
    gridResolutionX = $state(256);
    gridResolutionY = $state(256);
    gridResolutionZ = $state(96);
    simulationTimestepS = $state(1 / 144);

    renderMethodType = $state(GpuRenderMethodType.Points);

    readonly orbit = new CameraOrbit();
    readonly camera = new Camera({
        controlScheme: this.orbit,
        screenDims: { width: () => this.width, height: () => this.height },
    });

    readonly elapsedTime = new ElapsedTime();


    private device: GPUDevice | null = null;


    private stopSimulation = $state<(() => void) | null>(null);
    private runner = $state<GpuSnowPipelineRunner | null>(null);


    private onStatusChange: ((status: string) => void) | null = null;
    private onErr: ((err: string) => void) | null = null;


    constructor({
        onStatusChange = null,
        onErr = null,
    }: {
        onStatusChange?: ((status: string) => void) | null,
        onErr?: ((err: string) => void) | null,
    }) {
        this.onStatusChange = onStatusChange;
        this.onErr = onErr;
    }


    async reset() {
        if (this.runner === null || this.device === null) return;

        this.stopSimulation?.();
        this.stopSimulation = null;
        
        this.runner.scatterParticlesInMeshVolume();

        this.onStatusChange?.("initializing particles");

        await this.device.queue.onSubmittedWorkDone(); // need this to set simulation start time accurately
        if (this.stopSimulation !== null) return;

        this.onStatusChange?.("off and racing");

        this.stopSimulation = this.runner.loop({
            onAnimationFrameTimeUpdate: (ms) =>
                (this.elapsedTime.animationFrameTimeNs = BigInt(
                    Math.round(ms * 1_000_000),
                )),
            onGpuTimeUpdate: (ns) => (this.elapsedTime.gpuTimeNs = ns),
        });
    }


    static loadOntoCanvas({
        canvasPromise,
        onStatusChange,
        onErr,
    }: {
        canvasPromise: Promise<HTMLCanvasElement>,
        onStatusChange?: (status: string) => void,
        onErr?: (err: string) => void,
    }) {
        const state = new SimulationState({
            onStatusChange,
            onErr,
        });



        onMount(async () => {
            const response = await requestGpuDeviceAndContext({
                onStatusChange,
                onErr,
                canvas: await canvasPromise,
            });
            if (response === null) return;
            const { device, context, format, supportsTimestamp } = response;
            state.device = device;

            onStatusChange?.("loading geometry...");
            const { vertices } = await loadGltfScene(modelUrl);

            const { positions } = await loadGltfScene(colliderUrl);
            const { normals } = await loadGltfScene(colliderUrl);
            const { indices } = await loadGltfScene(colliderUrl); 

            const collider: ColliderGeometry = {
                positions,
                normals,
                indices,
            };

            state.width = innerWidth;
            state.height = innerHeight;

            state.runner = new GpuSnowPipelineRunner({
                device,
                format,
                context,
                nParticles: state.nParticles,
                gridResolutionX: state.gridResolutionX,
                gridResolutionY: state.gridResolutionY,
                gridResolutionZ: state.gridResolutionZ,
                simulationTimestepS: state.simulationTimestepS,
                camera: state.camera,
                meshVertices: vertices,
                collider: collider,
                getRenderMethodType: () => state.renderMethodType,
                measurePerf: supportsTimestamp,
            });

            state.reset();
        });

        onDestroy(() => {
            state.stopSimulation?.();
        });


        $effect(() => {
            state.runner?.resizeTextures(state.width, state.height);
        });


        return state;
    }
}