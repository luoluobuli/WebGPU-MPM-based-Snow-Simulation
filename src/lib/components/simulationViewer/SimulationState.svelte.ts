import { mat4 } from "wgpu-matrix";
import { onDestroy, onMount } from "svelte";
import { GpuSnowPipelineRunner } from "../../gpu/GpuSnowPipelineRunner.svelte";
import { requestGpuDeviceAndContext } from "../../gpu/requestGpuDeviceAndContext";
import { loadGltfScene } from "./loadScene";
import modelUrl from "$lib/assets/models/snow2.glb?url";
import colliderUrl from "$lib/assets/models/forest_scaled.glb?url";
import { CameraOrbit } from "./CameraOrbit.svelte";
import { Camera } from "./Camera.svelte";
import { ElapsedTime } from "./ElapsedTime.svelte";
import { GpuRenderMethodType } from "$lib/gpu/GpuRenderMethod";
import type { ColliderGeometry } from "../../gpu/collider/GpuColliderBufferManager";
import { GpuSimulationMethodType } from "$lib/gpu/GpuSimulationMethod";
import { loadEnvironmentMap } from "$lib/gpu/environmentMap/loadEnvironmentMap";
import { ParticleControlMode } from "./ParticleControlMode";

export class SimulationState {
    width = $state(300);
    height = $state(150);

    nParticles = $state(300_000);
    gridResolutionX = $state(256);
    gridResolutionY = $state(256);
    gridResolutionZ = $state(256);
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
    particleControlMode = $state(ParticleControlMode.Repel);


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
    interactionDistance = $state(15);
    interactionRadiusFactor = $state(3);
    interactionStrength = $state(1_500);
    interactionRadiusVal = $derived(this.interactionDistance * this.interactionRadiusFactor);

    colliderFriction = $state(0.25);

    onInteractionStart(x: number, y: number, el: HTMLElement) {
        this.isInteracting = true;
        this.updateInteractionRay(x, y, el, true);
    }

    onInteractionDrag(x: number, y: number, el: HTMLElement) {
        if (!this.isInteracting) return;
        this.updateInteractionRay(x, y, el, false);
    }

    onInteractionEnd() {
        this.isInteracting = false;
        this.runner?.uniformsManager.writeIsInteracting(false);
    }

 

    async updateInteractionRay(x: number, y: number, el: HTMLElement, isPointerDown: boolean) {
        if (!this.runner) return;

        const rect = el.getBoundingClientRect();
        
        // NDC
        const ndcX = ((x - rect.left) / rect.width) * 2 - 1;
        const ndcY = 1 - ((y - rect.top) / rect.height) * 2; 
        
        // Ray generation
        const invViewProj = this.camera.viewProjInvMat;
        
        const near = this.unproject(ndcX, ndcY, 0.0, invViewProj);
        const far = this.unproject(ndcX, ndcY, 1.0, invViewProj);
        
        const dir = [far[0] - near[0], far[1] - near[1], far[2] - near[2]];
        const len = Math.sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
        const dirNorm = [dir[0]/len, dir[1]/len, dir[2]/len];
        
        const minC = -5;
        const maxC = 5;
        const range = maxC - minC;
        const res = this.gridResolutionX; 
        
        if (isPointerDown) {
             let t = 20; 

             // Depth Picking (Current Surface)
             const px = (x - rect.left) * (this.width / rect.width);
             const py = (y - rect.top) * (this.height / rect.height);
             
             const depth = await this.runner.pickDepth(px, py);

             if (depth !== null && depth < 1.0) {
                 // DEPTH UNPROJECT
                 // We have NDC Z = depth.
                 // We have NDC X, Y.
                 // Unproject gives World Pos directly.
                 const worldHit = this.unproject(ndcX, ndcY, depth, invViewProj);
                 
                 // Distance from Near Plane to World Hit?
                 // Or just use worldHit directly?
                 // My logic uses interactionDistance along dirNorm from near.
                 // t = distance(near, worldHit).
                 const distVec = [worldHit[0] - near[0], worldHit[1] - near[1], worldHit[2] - near[2]];
                 t = Math.sqrt(distVec[0]*distVec[0] + distVec[1]*distVec[1] + distVec[2]*distVec[2]);
                 
                 // If t is weird, fallback?
             } else {
                 // Fallback to Plane Z=0 if miss
                 let tPlane = -near[2] / dirNorm[2];
                 if (tPlane > 0 && isFinite(tPlane)) t = tPlane;
             }

             this.interactionDistance = t;
        }
        
        const worldPos = [
            near[0] + dirNorm[0] * this.interactionDistance,
            near[1] + dirNorm[1] * this.interactionDistance,
            near[2] + dirNorm[2] * this.interactionDistance
        ];

        // Convert World Pos to Grid Pos
        const gridX = ((worldPos[0] - minC) / range) * res;
        const gridY = ((worldPos[1] - minC) / range) * res;
        const gridZ = ((worldPos[2] - minC) / range) * res;
        
        this.runner.uniformsManager.writeInteractionPos([gridX, gridY, gridZ]);
        this.runner.uniformsManager.writeInteractionDir(dirNorm as [number, number, number]);
        this.runner.uniformsManager.writeInteractionStrength(this.interactionStrength);
        this.runner.uniformsManager.writeInteractionRadius(this.interactionRadiusVal);
        this.runner.uniformsManager.writeInteractionMode(this.particleControlMode); 
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
                colliderFriction: () => state.colliderFriction,
            });

            state.restart();
        });

        onDestroy(() => {
            state.stopSimulation?.();
        });


        return state;
    }
}