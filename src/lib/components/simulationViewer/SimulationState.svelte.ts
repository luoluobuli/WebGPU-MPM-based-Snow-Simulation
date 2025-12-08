import { mat4 } from "wgpu-matrix";
import { onDestroy, onMount } from "svelte";
import { GpuSnowPipelineRunner } from "../../gpu/GpuSnowPipelineRunner.svelte";
import { requestGpuDeviceAndContext } from "../../gpu/requestGpuDeviceAndContext";
import { loadGltfScene } from "./loadScene";
import modelUrl from "$lib/assets/models/horse_statue_01_1k.glb?url";
import colliderUrl from "$lib/assets/models/forest.glb?url";
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

    nParticles = $state(15_000);
    gridResolutionX = $state(128);
    gridResolutionY = $state(128);
    gridResolutionZ = $state(128);
    explicitMpmSimulationTimestepS = $state(1 / 192);
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
    renderMethodType = $state(GpuRenderMethodType.MarchingCubes);


    readonly orbit = new CameraOrbit();
    readonly camera = new Camera({
        controlScheme: this.orbit,
        screenDims: { width: () => this.width, height: () => this.height },
    });

    readonly elapsedTime = new ElapsedTime();


    private device: GPUDevice | null = null;


    private stopSimulation = $state<(() => void) | null>(null);
    private runner = $state<GpuSnowPipelineRunner | null>(null);
    prerenderElapsedTimes = $derived(this.runner?.prerenderElapsedTimes ?? null);

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


    async restart() {
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
                this.elapsedTime.gpuRenderTimeNs = times.renderNs;
                this.elapsedTime.gpuPostprocessRenderTimeNs = times.postprocessRenderNs;
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

    isInteracting = $state(false);
    interactionPos = $state<[number, number, number]>([0, 0, 0]);
    readonly interactionRadiusVal = 20;
    readonly interactionStrength = 2000; // positive is repulsive

    onInteractionStart(event: PointerEvent) {
        this.isInteracting = true;
        this.updateInteractionRay(event);
    }

    onInteractionDrag(event: PointerEvent) {
        if (!this.isInteracting) return;
        this.updateInteractionRay(event);
    }

    onInteractionEnd() {
        this.isInteracting = false;
        this.runner?.uniformsManager.writeIsInteracting(false);
    }

    updateInteractionRay(event: PointerEvent) {
        if (!this.runner) return;

        const x = event.clientX;
        const y = event.clientY;
        const rect = (event.target as HTMLElement).getBoundingClientRect();
        
        // NDC
        const ndcX = ((x - rect.left) / rect.width) * 2 - 1;
        const ndcY = 1 - ((y - rect.top) / rect.height) * 2; // WebGPU Y is up in NDC? No, -1 to 1. HTML Y is down.
        
        // Ray generation
        const invViewProj = this.camera.viewProjInvMat;
        
        const near = this.unproject(ndcX, ndcY, 0.0, invViewProj);
        const far = this.unproject(ndcX, ndcY, 1.0, invViewProj);
        
        const dir = [far[0] - near[0], far[1] - near[1], far[2] - near[2]];
        const len = Math.sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
        const dirNorm = [dir[0]/len, dir[1]/len, dir[2]/len];
        

        // grid space conversion
        const minC = -5;
        const maxC = 5;
        const range = maxC - minC;
        const res = this.gridResolutionX; 


        // near plane
        const gridOriginX = ((near[0] - minC) / range) * res;
        const gridOriginY = ((near[1] - minC) / range) * res;
        const gridOriginZ = ((near[2] - minC) / range) * res;
        
        this.runner.uniformsManager.writeInteractionPos([gridOriginX, gridOriginY, gridOriginZ]);
        this.runner.uniformsManager.writeInteractionDir(dirNorm as [number, number, number]);
        this.runner.uniformsManager.writeInteractionStrength(this.interactionStrength);
        this.runner.uniformsManager.writeInteractionRadius(this.interactionRadiusVal);
        this.runner.uniformsManager.writeIsInteracting(true);
    }

    private unproject(x: number, y: number, z: number, invMat: Float32Array): [number, number, number] {
        const v = [x, y, z, 1.0];
        const out = [0,0,0,0];
        out[0] = invMat[0]*v[0] + invMat[4]*v[1] + invMat[8]*v[2] + invMat[12]*v[3];
        out[1] = invMat[1]*v[0] + invMat[5]*v[1] + invMat[9]*v[2] + invMat[13]*v[3];
        out[2] = invMat[2]*v[0] + invMat[6]*v[1] + invMat[10]*v[2] + invMat[14]*v[3];
        out[3] = invMat[3]*v[0] + invMat[7]*v[1] + invMat[11]*v[2] + invMat[15]*v[3];
        
        return [out[0]/out[3], out[1]/out[3], out[2]/out[3]];
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
            const { positions, normals, uvs, materialIndices, textures, indices, objects } = await loadGltfScene(colliderUrl);

            const collider: ColliderGeometry = {
                positions,
                normals,
                uvs,
                materialIndices,
                textures,
                indices,
                objects,
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
                width: () => state.width,
                height: () => state.height,
            });

            state.restart();
        });

        onDestroy(() => {
            state.stopSimulation?.();
        });


        return state;
    }
}