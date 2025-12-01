import { mat4 } from "wgpu-matrix";
import { onDestroy, onMount } from "svelte";
import { GpuSnowPipelineRunner } from "../../gpu/GpuSnowPipelineRunner.svelte";
import { requestGpuDeviceAndContext } from "../../gpu/requestGpuDeviceAndContext";
import { loadGltfScene } from "./loadScene";
import modelUrl from "$lib/assets/models/monkey.glb?url";
import colliderUrl from "$lib/assets/models/test2.glb?url";
import { CameraOrbit } from "./CameraOrbit.svelte";
import { Camera } from "./Camera.svelte";
import { ElapsedTime } from "./ElapsedTime.svelte";
import { GpuRenderMethodType } from "$lib/gpu/GpuRenderMethod";
import type { ColliderGeometry } from "../../gpu/collider/GpuColliderBufferManager";
import { GpuSimulationMethodType } from "$lib/gpu/GpuSimulationMethod";
import { loadEnvironmentMap } from "$lib/gpu/environmentMap/loadEnvironmentMap";

export class SimulationState {
    width = $state(300);
    height = $state(150);

    nParticles = $state(500_000);
    gridResolutionX = $state(512);
    gridResolutionY = $state(512);
    gridResolutionZ = $state(192);
    explicitMpmSimulationTimestepS = $state(1 / 384);
    pbmpmSimulationTimestepS = $state(1 / 384);
    transformMat = $state(mat4.identity());

    oneSimulationStepPerFrame = $state(true);

    moveForward  = $state(false); // W
    moveBackward = $state(false); // S
    moveLeft     = $state(false); // A
    moveRight    = $state(false); // D
    moveUp       = $state(false); // Q
    moveDown     = $state(false); // E

    simulationMethodType = $state(GpuSimulationMethodType.Pbmpm);
    renderMethodType = $state(GpuRenderMethodType.Volumetric);

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
            onGpuTimeUpdate: (times) => {
                this.elapsedTime.gpuComputeSimulationStepTimeNs = times.computeSimulationStepNs;
                this.elapsedTime.gpuComputePrerenderTimeNs = times.computePrerenderNs;
                this.elapsedTime.gpuRenderTimeNs = times.renderNs;
            },
            onUserControlUpdate: () => {
                const speed = 0.02;
                this.runner?.updateColliderVel([0.0, 0.0, 0.0]);
                if (this.moveForward) { this.applyColliderTransform([0.0, -speed, 0.0]); }
                if (this.moveBackward) { this.applyColliderTransform([0.0, speed, 0.0]); }
                if (this.moveLeft) { this.applyColliderTransform([speed, 0.0, 0.0]); }
                if (this.moveRight) { this.applyColliderTransform([-speed, 0.0, 0.0]); }
                if (this.moveUp) { this.applyColliderTransform([0.0, 0.0, speed]); }
                if (this.moveDown) { this.applyColliderTransform([0.0, 0.0, -speed]); }
            },
        });
    }

    applyColliderTransform(step: [number, number, number]) {
        const t = mat4.translation(step);
        this.transformMat = mat4.mul(t, this.transformMat);
        this.runner?.updateColliderTransformMat(this.transformMat);
        this.runner?.updateColliderVel(step);
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
            const { positions, normals, indices } = await loadGltfScene(colliderUrl);

            const collider: ColliderGeometry = {
                positions,
                normals,
                indices,
                //transform: state.transformMat,
            };

            onStatusChange?.("loading environment...");
            const environmentImageBitmap = await loadEnvironmentMap();

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
                explicitMpmSimulationTimestepS: () => state.explicitMpmSimulationTimestepS,
                pbmpmSimulationTimestepS: () => state.pbmpmSimulationTimestepS,
                camera: state.camera,
                meshVertices: vertices,
                collider: collider,
                getSimulationMethodType: () => state.simulationMethodType,
                getRenderMethodType: () => state.renderMethodType,
                oneSimulationStepPerFrame: () => state.oneSimulationStepPerFrame,
                environmentImageBitmap,
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