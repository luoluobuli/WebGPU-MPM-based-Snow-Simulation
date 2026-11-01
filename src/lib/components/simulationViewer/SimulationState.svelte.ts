import { onDestroy, onMount } from "svelte";
import {
    GpuSnowPipelineRunner,
    type GpuFrameTiming,
} from "../../gpu/GpuSnowPipelineRunner.svelte";
import { requestGpuDeviceAndContext } from "../../gpu/requestGpuDeviceAndContext";
import { loadGltfScene } from "./loadScene";
import { CameraOrbit } from "./CameraOrbit.svelte";
import { Camera } from "./Camera.svelte";
import { ElapsedTime } from "./ElapsedTime.svelte";
import { GpuRenderMethodType } from "$lib/gpu/GpuRenderMethod";
import type { ColliderGeometry } from "../../gpu/collider/GpuColliderBufferManager";
import { GpuSimulationMethodType } from "$lib/gpu/GpuSimulationMethod";
import { loadEnvironmentMap } from "$lib/gpu/environmentMap/loadEnvironmentMap";
import { ParticleControlMode } from "./ParticleControlMode";
import {
    defaultSimulationScene,
    type SimulationSceneConfig,
} from "./SimulationScene";
import type { SpawnMeshSource } from "$lib/gpu/particleInitialize/GpuSpawnVolumeBufferManager";
import { vec3 } from "wgpu-matrix";


const errorToString = (error: unknown) => error instanceof Error ? error.message : String(error);

const loadSpawnSource = async (
    scene: SimulationSceneConfig,
): Promise<{
    spawnSource: SpawnMeshSource,
}> => {
    const { vertices, objects } = await loadGltfScene(scene.spawnSource.url);

    return {
        spawnSource: {
            vertices,
            objects,
        },
    };
};

const loadCollider = async (
    scene: SimulationSceneConfig,
): Promise<ColliderGeometry | null> => {
    if (scene.colliderSource === null) {
        return null;
    }

    const {
        positions,
        normals,
        uvs,
        materialIndices,
        textures,
        indices,
        objects,
    } = await loadGltfScene(scene.colliderSource.url);

    return { positions, normals, uvs, materialIndices, textures, indices, objects };
};

export class SimulationState {
    width = $state(300);
    height = $state(150);

    nParticles = $state(300_000);
    gridResolutionX = $state(384);
    gridResolutionY = $state(384);
    gridResolutionZ = $state(384);
    explicitMpmMaxSimulationTimestepS = $state(1 / 192);
    mlsMpmMaxSimulationTimestepS = $state(1 / 1024);
    pbmpmMaxSimulationTimestepS = $state(1 / 240);
    pbmpmSolveIterations = $state(4);

    oneSimulationStepPerFrame = $state(true);

    simulationMethodType = $state(GpuSimulationMethodType.MlsMpm);
    renderMethodType = $state(GpuRenderMethodType.Splats);
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
    actualSimulationTimestepS = $derived(this.runner?.selectedSimulationTimestepS ?? null);
    private restartEpoch = 0;

    private onStatusChange: ((status: string) => void) | null = null;
    private onErr: ((err: string) => void) | null = null;
    private readonly scene: SimulationSceneConfig;


    constructor({
        scene = defaultSimulationScene,
        onStatusChange = null,
        onErr = null,
    }: {
        scene?: SimulationSceneConfig,
        onStatusChange?: ((status: string) => void) | null,
        onErr?: ((err: string) => void) | null,
    }) {
        this.scene = scene;
        this.onStatusChange = onStatusChange;
        this.onErr = onErr;
        this.nParticles = scene.nParticles;
        this.gridResolutionX = scene.gridResolution[0];
        this.gridResolutionY = scene.gridResolution[1];
        this.gridResolutionZ = scene.gridResolution[2];
        this.simulationMethodType = scene.simulationMethodType;
        this.renderMethodType = scene.renderMethodType;
        if (scene.timing?.explicitMpmMaxSimulationTimestepS !== undefined) {
            this.explicitMpmMaxSimulationTimestepS = scene.timing.explicitMpmMaxSimulationTimestepS;
        }
        if (scene.timing?.mlsMpmMaxSimulationTimestepS !== undefined) {
            this.mlsMpmMaxSimulationTimestepS = scene.timing.mlsMpmMaxSimulationTimestepS;
        }
        if (scene.timing?.pbmpmMaxSimulationTimestepS !== undefined) {
            this.pbmpmMaxSimulationTimestepS = scene.timing.pbmpmMaxSimulationTimestepS;
        }
        if (scene.timing?.pbmpmSolveIterations !== undefined) {
            this.pbmpmSolveIterations = scene.timing.pbmpmSolveIterations;
        }
        if (scene.timing?.oneSimulationStepPerFrame !== undefined) {
            this.oneSimulationStepPerFrame = scene.timing.oneSimulationStepPerFrame;
        }
        if (scene.colliderFriction !== undefined) {
            this.colliderFriction = scene.colliderFriction;
        }

        if (scene.camera?.radius !== undefined) {
            this.orbit.radius = scene.camera.radius;
        }
        if (scene.camera?.lat !== undefined) {
            this.orbit.lat = scene.camera.lat;
        }
        if (scene.camera?.long !== undefined) {
            this.orbit.long = scene.camera.long;
        }
        if (scene.camera?.offset !== undefined) {
            this.orbit.offset = vec3.fromValues(...scene.camera.offset);
        }
    }

    private updateAnimationFrameTime(ms: number) {
        this.elapsedTime.animationFrameTimeNs = BigInt(
            Math.round(ms * 1_000_000),
        );
    }

    private updateGpuTime(times: GpuFrameTiming) {
        this.elapsedTime.gpuComputeSimulationStepTimeNs = times.computeSimulationStepNs;
        this.elapsedTime.gpuComputeSimulationSubstepTimeNs = times.computeSimulationSubstepNs;
        this.elapsedTime.nSimulationSubsteps = times.nSimulationSubsteps;
        this.elapsedTime.gpuRenderTimeNs = times.renderNs;
        this.elapsedTime.gpuPostprocessRenderTimeNs = times.postprocessRenderNs;
    }

    private readonly frameTimingCallbacks = {
        onAnimationFrameTimeUpdate: (ms: number) => this.updateAnimationFrameTime(ms),
        onGpuTimeUpdate: (times: GpuFrameTiming) => this.updateGpuTime(times),
    };

    setViewportWidth(width: number | null | undefined) {
        this.setViewportSize({ width });
    }

    setViewportHeight(height: number | null | undefined) {
        this.setViewportSize({ height });
    }

    private setViewportSize({
        width = this.width,
        height = this.height,
    }: {
        width?: number | null | undefined,
        height?: number | null | undefined,
    }) {
        const nextWidth = typeof width === "number" && Number.isFinite(width)
            ? Math.max(1, Math.floor(width))
            : this.width;
        const nextHeight = typeof height === "number" && Number.isFinite(height)
            ? Math.max(1, Math.floor(height))
            : this.height;

        if (nextWidth === this.width && nextHeight === this.height) return;

        this.width = nextWidth;
        this.height = nextHeight;
    }

    turnCamera(movement: { x: number, y: number }) {
        this.orbit.turn(movement);
    }

    panCamera(movement: { x: number, y: number }) {
        this.orbit.pan(movement);
    }

    zoomCamera(deltaY: number) {
        this.orbit.radius *= 2 ** (deltaY * 0.001);
    }


    async restart() {
        if (this.runner === null || this.device === null) return;

        const restartEpoch = ++this.restartEpoch;

        this.stopSimulation?.();
        this.stopSimulation = null;

        this.onStatusChange?.("initializing particles...");
        
        try {
            this.runner.scatterParticles();

            await this.device.queue.onSubmittedWorkDone(); // need this to set simulation start time accurately

            if (restartEpoch !== this.restartEpoch) return;
        } catch (error) {
            console.error(error);
            this.onErr?.(errorToString(error));
            return;
        }

        if (restartEpoch !== this.restartEpoch || this.stopSimulation !== null) return;

        this.onStatusChange?.("off and racing");

        this.stopSimulation = this.runner.loop(this.frameTimingCallbacks);
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
        
        const domainMin = this.scene.domain?.min ?? [-5, -5, -5];
        const domainMax = this.scene.domain?.max ?? [5, 5, 5];
        const gridResolution = [
            this.gridResolutionX,
            this.gridResolutionY,
            this.gridResolutionZ,
        ];
        
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
        const gridX = (
            (worldPos[0] - domainMin[0])
            / (domainMax[0] - domainMin[0])
        ) * gridResolution[0];
        const gridY = (
            (worldPos[1] - domainMin[1])
            / (domainMax[1] - domainMin[1])
        ) * gridResolution[1];
        const gridZ = (
            (worldPos[2] - domainMin[2])
            / (domainMax[2] - domainMin[2])
        ) * gridResolution[2];
        
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
        scene = defaultSimulationScene,
        canvasPromise,
        onStatusChange,
        onErr,
    }: {
        scene?: SimulationSceneConfig,
        canvasPromise: Promise<HTMLCanvasElement>,
        onStatusChange?: (status: string) => void,
        onErr?: (err: string) => void,
    }) {
        let destroyed = false;
        const updateStatus = (status: string) => {
            if (!destroyed) {
                onStatusChange?.(status);
            }
        };
        const updateErr = (err: string) => {
            if (!destroyed) {
                onErr?.(err);
            }
        };

        const state = new SimulationState({
            scene,
            onStatusChange: updateStatus,
            onErr: updateErr,
        });



        onMount(() => {
            void (async () => {
                try {
                    const canvas = await canvasPromise;
                    if (destroyed) return;

                    const response = await requestGpuDeviceAndContext({
                        onStatusChange: updateStatus,
                        onErr: updateErr,
                        canvas,
                    });
                    if (response === null) return;
                    const { device, context, format, supportsTimestamp } = response;
                    if (destroyed) {
                        device.destroy();
                        return;
                    }
                    state.device = device;

                    updateStatus("loading particles...");
                    const { spawnSource } = await loadSpawnSource(scene);
                    if (destroyed) return;

                    updateStatus("loading collider...");
                    const collider = await loadCollider(scene);
                    if (destroyed) return;

                    updateStatus("loading environment...");
                    const environmentImageBitmap = await loadEnvironmentMap();
                    if (destroyed) return;

                    updateStatus("initializing renderer...");
                    if (destroyed) return;

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
                        domain: scene.domain,
                        explicitMpmMaxSimulationTimestepS: () => state.explicitMpmMaxSimulationTimestepS,
                        mlsMpmMaxSimulationTimestepS: () => state.mlsMpmMaxSimulationTimestepS,
                        pbmpmMaxSimulationTimestepS: () => state.pbmpmMaxSimulationTimestepS,
                        pbmpmSolveIterations: () => state.pbmpmSolveIterations,
                        camera: state.camera,
                        spawnSource,
                        collider,
                        getSimulationMethodType: () => state.simulationMethodType,
                        getRenderMethodType: () => state.renderMethodType,
                        oneSimulationStepPerFrame: () => state.oneSimulationStepPerFrame,
                        environmentImageBitmap,
                        measurePerf: supportsTimestamp,
                        width: () => state.width,
                        height: () => state.height,
                        colliderFriction: () => state.colliderFriction,
                        isInteracting: () => state.isInteracting,
                        interactionStrength: () => state.interactionStrength,
                    });

                    await state.restart();
                } catch (error) {
                    console.error(error);
                    updateErr(errorToString(error));
                }
            })();
        });

        onDestroy(() => {
            destroyed = true;
            state.restartEpoch++;
            state.stopSimulation?.();
            state.stopSimulation = null;
            state.runner?.destroy();
            state.runner = null;
            state.device?.destroy();
            state.device = null;
        });


        return state;
    }
}
