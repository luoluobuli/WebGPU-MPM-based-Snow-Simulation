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
import { buildProceduralForest } from "./proceduralForest";
import { loadTreeModelSpawnPoints } from "./loadTreeModel";
import type { SpawnPointSource } from "$lib/gpu/particleInitialize/GpuSpawnVolumeBufferManager";
import { vec3 } from "wgpu-matrix";
import {
    canSelectSimulationFrameCacheDirectory,
    createSimulationFrameFileCache,
    isQuotaExceededError,
    selectSimulationFrameCacheDirectory,
    type SimulationFrameFileCache,
} from "./SimulationFrameFileCache";
import {
    TIMELINE_DEFAULT_SECONDS_PER_FRAME,
    TIMELINE_FRAME_COUNT,
    TIMELINE_FRAME_INTERVAL_MS,
    clampTimelineSecondsPerFrame,
    calculateTimelineBakeSubsteps,
    timelineSecondsPerFrameFromStepDivisor,
    timelineStepDivisorFromSecondsPerFrame,
    timelineFrameSimulationTimeS,
} from "./SimulationTimelineTiming";

const TIMELINE_BACKGROUND_WORK_YIELD_MS = 250;

const waitForTimelineWorkYield = () => new Promise<void>((resolve) => {
    if (typeof requestAnimationFrame === "undefined") {
        setTimeout(resolve, 0);
        return;
    }

    let resolved = false;
    const timeoutHandle = setTimeout(
        () => {
            if (resolved) return;

            resolved = true;
            resolve();
        },
        TIMELINE_BACKGROUND_WORK_YIELD_MS,
    );

    const finish = () => {
        if (resolved) return;

        resolved = true;
        clearTimeout(timeoutHandle);
        resolve();
    };

    requestAnimationFrame(() => requestAnimationFrame(finish));
});

const errorToString = (error: unknown) => error instanceof Error ? error.message : String(error);

const formatCacheBytes = (bytes: number | null) => {
    if (bytes === null || !Number.isFinite(bytes)) return "unknown";

    const absBytes = Math.abs(bytes);
    if (absBytes >= 1024 * 1024 * 1024) {
        return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GiB`;
    }

    return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
};

const sleep = (ms: number) => new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
});

const hashString = (text: string) => {
    let hash = 0x811c9dc5;

    for (let i = 0; i < text.length; i++) {
        hash ^= text.charCodeAt(i);
        hash = Math.imul(hash, 0x01000193);
    }

    return (hash >>> 0).toString(16).padStart(8, "0");
};

const loadSpawnSource = async (
    scene: SimulationSceneConfig,
    nParticles: number,
): Promise<{
    spawnSource: SpawnPointSource,
    particleAppearances: Uint32Array | null,
}> => {
    switch (scene.spawnSource.type) {
        case "mesh": {
            const { vertices, objects } = await loadGltfScene(scene.spawnSource.url);

            return {
                spawnSource: {
                    type: "mesh",
                    vertices,
                    objects,
                },
                particleAppearances: null,
            };
        }

        case "treeModel": {
            const tree = await loadTreeModelSpawnPoints({
                url: scene.spawnSource.url,
                nParticles,
            });

            return {
                spawnSource: {
                    type: "points",
                    points: tree.spawnPoints,
                },
                particleAppearances: tree.particleAppearances,
            };
        }

        case "proceduralForest": {
            const forest = buildProceduralForest({
                nParticles,
                seed: scene.spawnSource.seed,
            });

            return {
                spawnSource: {
                    type: "points",
                    points: forest.spawnPoints,
                },
                particleAppearances: forest.particleAppearances,
            };
        }
    }
};

const loadCollider = async (
    scene: SimulationSceneConfig,
): Promise<ColliderGeometry | null> => {
    if (scene.colliderSource === null) {
        return null;
    }

    const { positions, normals, uvs, materialIndices, textures, indices, objects } = await loadGltfScene(scene.colliderSource.url);

    return {
        positions,
        normals,
        uvs,
        materialIndices,
        textures,
        indices,
        objects,
    };
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
    readonly timeline: boolean;

    timelineFrameCount = $state(TIMELINE_FRAME_COUNT);
    timelineFrame = $state(0);
    timelineCachedThroughFrame = $state(-1);
    timelineCachedFrameCount = $derived(Math.max(0, this.timelineCachedThroughFrame + 1));
    timelineIsBusy = $state(false);
    timelineIsPlaying = $state(false);
    timelineStorageLabel = $state("file cache pending");
    timelineStatus = $state("waiting for renderer");
    timelineCanSelectCacheDirectory = canSelectSimulationFrameCacheDirectory();
    timelineFrameByteLength = $derived(this.runner?.simulationPlaybackFrameLayout.byteLength ?? null);
    timelineSecondsPerFrame = $state(TIMELINE_DEFAULT_SECONDS_PER_FRAME);
    timelineStepDivisor = $derived(
        timelineStepDivisorFromSecondsPerFrame(this.timelineSecondsPerFrame),
    );
    timelineDurationS = $derived(
        Math.max(0, this.timelineFrameCount - 1) * this.timelineSecondsPerFrame,
    );

    private timelineCache: SimulationFrameFileCache | null = null;
    private timelineCacheRootDirectory: FileSystemDirectoryHandle | null = null;
    private timelineCacheKey = "";
    private timelineRunToken = 0;
    private timelineSimulatedTimeS = 0;
    private timelineSolverFrame: number | null = 0;
    private stillFrameRenderHandle: number | null = null;
    private stillFrameRenderRequested = false;
    private stillFrameRenderInFlight = false;


    constructor({
        scene = defaultSimulationScene,
        timeline = false,
        onStatusChange = null,
        onErr = null,
    }: {
        scene?: SimulationSceneConfig,
        timeline?: boolean,
        onStatusChange?: ((status: string) => void) | null,
        onErr?: ((err: string) => void) | null,
    }) {
        this.scene = scene;
        this.timeline = timeline;
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
        if (scene.timing?.oneSimulationStepPerFrame !== undefined) {
            this.oneSimulationStepPerFrame = scene.timing.oneSimulationStepPerFrame;
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

    private currentTimelineCacheKey() {
        return hashString(JSON.stringify({
            scene: this.scene,
            nParticles: this.nParticles,
            gridResolution: [
                this.gridResolutionX,
                this.gridResolutionY,
                this.gridResolutionZ,
            ],
            simulationMethodType: this.simulationMethodType,
            explicitMpmMaxSimulationTimestepS: this.explicitMpmMaxSimulationTimestepS,
            mlsMpmMaxSimulationTimestepS: this.mlsMpmMaxSimulationTimestepS,
            timelineFrameCount: TIMELINE_FRAME_COUNT,
            timelineSecondsPerFrame: this.timelineSecondsPerFrame,
            timelineFrameFormat: "position-material-v1",
        }));
    }

    private async updateTimelineCacheCapacity() {
        if (this.runner === null || this.timelineCache === null) return true;

        const frameByteLength = this.runner.simulationPlaybackFrameLayout.byteLength;
        const capacity = await this.timelineCache.estimateFrameCapacity({
            frameByteLength,
            requestedFrameCount: TIMELINE_FRAME_COUNT,
        });
        const cappedFrameCount = Math.min(
            TIMELINE_FRAME_COUNT,
            capacity.frameCount,
        );

        if (cappedFrameCount <= 0) {
            this.timelineFrameCount = TIMELINE_FRAME_COUNT;
            this.timelineStorageLabel = `${this.timelineCache.storageLabel} (capacity estimate low)`;
            this.timelineStatus = `browser storage estimate is below one ${formatCacheBytes(frameByteLength)} cache frame; caching until a write fails`;

            return true;
        }

        this.timelineFrameCount = TIMELINE_FRAME_COUNT;
        this.timelineStorageLabel = cappedFrameCount < TIMELINE_FRAME_COUNT
            ? `${this.timelineCache.storageLabel} (storage estimate ${cappedFrameCount}/${TIMELINE_FRAME_COUNT} frames)`
            : this.timelineCache.storageLabel;

        if (cappedFrameCount < TIMELINE_FRAME_COUNT) {
            this.timelineStatus = `browser storage estimate allows ${cappedFrameCount}/${TIMELINE_FRAME_COUNT} frames; caching until a write fails`;
        }

        return true;
    }

    private async resetTimelineCache() {
        const cacheKey = this.currentTimelineCacheKey();
        this.timelineFrameCount = TIMELINE_FRAME_COUNT;
        this.timelineSolverFrame = 0;

        if (this.timelineCache === null || this.timelineCacheKey !== cacheKey) {
            this.timelineCache = await createSimulationFrameFileCache({
                cacheKey,
                rootDirectory: this.timelineCacheRootDirectory,
            });
            this.timelineCacheKey = cacheKey;
            this.timelineStorageLabel = this.timelineCache.storageLabel;
        }

        await this.timelineCache.clear();
        this.timelineCachedThroughFrame = -1;

        return await this.updateTimelineCacheCapacity();
    }

    private async cacheCurrentTimelineFrame(frameIndex: number) {
        if (this.runner === null || this.timelineCache === null) return;

        const frame = await this.runner.readSimulationPlaybackFrame();
        await this.timelineCache.writeFrame(frameIndex, frame);
        this.timelineCachedThroughFrame = Math.max(
            this.timelineCachedThroughFrame,
            frameIndex,
        );
    }

    private async restoreTimelineFrameFromCache(
        frameIndex: number,
        {
            render = true,
        }: {
            render?: boolean,
        } = {},
    ) {
        if (this.runner === null || this.timelineCache === null) return false;

        const frame = await this.timelineCache.readFrame(frameIndex);
        if (frame === null) {
            return false;
        }

        this.runner.restoreSimulationPlaybackFrame(frame);
        this.timelineSolverFrame = null;
        this.timelineSimulatedTimeS = timelineFrameSimulationTimeS(
            frameIndex,
            this.timelineSecondsPerFrame,
        );
        if (render) {
            this.runner.renderStillFrame(this.frameTimingCallbacks);
        }

        this.timelineFrame = frameIndex;
        return true;
    }

    private async rebuildTimelineSolverThroughFrame(
        targetFrame: number,
        runToken: number,
    ) {
        if (this.runner === null) return false;
        if (this.timelineSolverFrame === targetFrame) return true;

        this.timelineStatus = `rebuilding solver state at frame ${targetFrame}...`;
        this.runner.scatterParticles();
        this.timelineSolverFrame = 0;
        this.timelineSimulatedTimeS = 0;
        await waitForTimelineWorkYield();
        if (runToken !== this.timelineRunToken) return false;

        for (let frameIndex = 1; frameIndex <= targetFrame; frameIndex++) {
            const nSubsteps = calculateTimelineBakeSubsteps({
                currentSimulationTimeS: this.timelineSimulatedTimeS,
                targetFrame: frameIndex,
                simulationTimestepS: this.runner.selectedSimulationTimestepS,
                secondsPerFrame: this.timelineSecondsPerFrame,
            });

            this.timelineStatus = `rebuilding solver frame ${frameIndex} (${nSubsteps} substeps)...`;
            const result = this.runner.advanceFixedSimulationSubsteps({
                nSubsteps,
                ...this.frameTimingCallbacks,
            });
            this.timelineSimulatedTimeS += result.simulatedTimeS;
            this.timelineSolverFrame = frameIndex;
            if (runToken !== this.timelineRunToken) return false;
        }

        return true;
    }

    private async initializeTimelineCache(restartEpoch: number) {
        if (this.runner === null) return;

        const runToken = ++this.timelineRunToken;
        this.timelineIsPlaying = false;
        this.timelineIsBusy = true;
        this.timelineFrame = 0;
        this.timelineSimulatedTimeS = 0;
        this.timelineStatus = "opening file cache...";

        try {
            const cacheHasRoom = await this.resetTimelineCache();
            if (restartEpoch !== this.restartEpoch || runToken !== this.timelineRunToken) return;
            if (!cacheHasRoom) {
                this.runner.renderStillFrame(this.frameTimingCallbacks);
                this.onStatusChange?.("timeline cache unavailable");

                return;
            }

            this.timelineStatus = "caching frame 0...";
            await waitForTimelineWorkYield();
            if (restartEpoch !== this.restartEpoch || runToken !== this.timelineRunToken) return;

            this.runner.renderStillFrame(this.frameTimingCallbacks);
            await this.cacheCurrentTimelineFrame(0);
            if (restartEpoch !== this.restartEpoch || runToken !== this.timelineRunToken) return;

            this.timelineFrame = 0;
            this.timelineSolverFrame = 0;
            this.timelineStatus = `cached frame 0 to ${this.timelineStorageLabel}`;
            this.onStatusChange?.("timeline ready");
        } catch (error) {
            if (isQuotaExceededError(error)) {
                await this.handleTimelineQuotaExceeded();
                return;
            }

            throw error;
        } finally {
            if (runToken === this.timelineRunToken) {
                this.timelineIsBusy = false;
            }
        }
    }

    private async bakeTimelineThroughFrame(
        targetFrame: number,
        runToken: number,
    ) {
        if (this.runner === null || this.timelineCache === null) return;

        if (this.timelineCachedThroughFrame >= 0) {
            const solverReady = await this.rebuildTimelineSolverThroughFrame(
                this.timelineCachedThroughFrame,
                runToken,
            );
            if (!solverReady) return;
        }

        for (
            let frameIndex = this.timelineCachedThroughFrame + 1;
            frameIndex <= targetFrame;
            frameIndex++
        ) {
            if (runToken !== this.timelineRunToken) return;

            const nSubsteps = calculateTimelineBakeSubsteps({
                currentSimulationTimeS: this.timelineSimulatedTimeS,
                targetFrame: frameIndex,
                simulationTimestepS: this.runner.selectedSimulationTimestepS,
                secondsPerFrame: this.timelineSecondsPerFrame,
            });

            this.timelineStatus = `simulating frame ${frameIndex} (${nSubsteps} substeps)...`;
            const result = this.runner.advanceFixedSimulationSubsteps({
                nSubsteps,
                ...this.frameTimingCallbacks,
            });
            this.timelineSimulatedTimeS += result.simulatedTimeS;
            this.timelineSolverFrame = frameIndex;
            if (runToken !== this.timelineRunToken) return;

            this.timelineStatus = `writing frame ${frameIndex}...`;
            await this.cacheCurrentTimelineFrame(frameIndex);
            if (runToken !== this.timelineRunToken) return;

            this.timelineFrame = frameIndex;
            this.timelineStatus = `cached frame ${frameIndex} to ${this.timelineStorageLabel}`;
            await waitForTimelineWorkYield();
        }
    }

    private async seekTimelineFrame(
        frameIndex: number,
        {
            keepPlaying = false,
            runToken = ++this.timelineRunToken,
        }: {
            keepPlaying?: boolean,
            runToken?: number,
        } = {},
    ) {
        if (!this.timeline || this.runner === null || this.timelineCache === null) return;

        const targetFrame = Math.max(
            0,
            Math.min(this.timelineFrameCount - 1, Math.round(frameIndex)),
        );

        if (!keepPlaying) {
            this.timelineIsPlaying = false;
        }

        this.timelineIsBusy = true;
        try {
            if (targetFrame <= this.timelineCachedThroughFrame) {
                this.timelineStatus = `restoring frame ${targetFrame}...`;
                const restored = await this.restoreTimelineFrameFromCache(targetFrame);
                if (!restored) {
                    throw new Error(`missing cached frame ${targetFrame}`);
                }
                this.timelineStatus = `restored frame ${targetFrame}`;
                return;
            }

            await this.bakeTimelineThroughFrame(targetFrame, runToken);
        } catch (error) {
            if (isQuotaExceededError(error)) {
                await this.handleTimelineQuotaExceeded();
                return;
            }

            console.error(error);
            const text = errorToString(error);
            this.timelineStatus = `error: ${text}`;
            this.timelineIsPlaying = false;
            this.onErr?.(text);
        } finally {
            if (runToken === this.timelineRunToken) {
                this.timelineIsBusy = false;
            }
        }
    }

    private async handleTimelineQuotaExceeded() {
        this.timelineIsPlaying = false;

        const cachedFrameCount = Math.max(0, this.timelineCachedThroughFrame + 1);
        this.timelineFrameCount = Math.max(1, cachedFrameCount);
        this.timelineStorageLabel = this.timelineCache === null
            ? "file cache full"
            : `${this.timelineCache.storageLabel} (${cachedFrameCount}/${TIMELINE_FRAME_COUNT} frames cached)`;

        if (this.timelineCachedThroughFrame >= 0) {
            await this.restoreTimelineFrameFromCache(this.timelineCachedThroughFrame);
            this.timelineStatus = `cache quota reached at frame ${cachedFrameCount}; using cached frames 0-${this.timelineCachedThroughFrame}`;
            this.onStatusChange?.("timeline cache full");
            return;
        }

        const frameByteLength = this.runner?.simulationPlaybackFrameLayout.byteLength ?? null;
        this.timelineStatus = `browser storage quota exceeded before frame 0 (${formatCacheBytes(frameByteLength)} / frame)`;
        this.onStatusChange?.("timeline cache unavailable");
    }

    async setTimelineFrame(frameIndex: number) {
        if (this.timelineIsBusy) return;

        await this.seekTimelineFrame(frameIndex);
    }

    async stepTimelineFrame(delta: number) {
        if (this.timelineIsBusy) return;

        await this.seekTimelineFrame(this.timelineFrame + delta);
    }

    pauseTimeline() {
        this.timelineRunToken++;
        this.timelineIsPlaying = false;
    }

    async setTimelineSecondsPerFrame(secondsPerFrame: number) {
        const clampedSecondsPerFrame = clampTimelineSecondsPerFrame(secondsPerFrame);
        if (Math.abs(clampedSecondsPerFrame - this.timelineSecondsPerFrame) < 1e-9) return;
        if (this.timelineIsBusy) return;

        this.pauseTimeline();
        this.timelineSecondsPerFrame = clampedSecondsPerFrame;
        this.timelineStatus = "timeline step changed; rebuilding cache...";

        if (!this.timeline || this.runner === null) return;

        await this.initializeTimelineCache(this.restartEpoch);
    }

    async setTimelineStepDivisor(stepDivisor: number) {
        await this.setTimelineSecondsPerFrame(
            timelineSecondsPerFrameFromStepDivisor(stepDivisor),
        );
    }

    async selectTimelineCacheDirectory() {
        if (!this.timeline || this.timelineIsBusy) return;

        if (!this.timelineCanSelectCacheDirectory) {
            this.timelineStatus = "folder cache requires browser directory picker support; using browser file cache";
            return;
        }

        this.pauseTimeline();
        this.timelineStatus = "selecting cache folder...";

        try {
            const directory = await selectSimulationFrameCacheDirectory();
            this.timelineCacheRootDirectory = directory;
            this.timelineCache = null;
            this.timelineCacheKey = "";
            this.timelineStorageLabel = `folder cache pending: ${directory.name || "selected folder"}`;
            this.timelineStatus = "cache folder selected; rebuilding cache...";

            if (this.runner === null) return;

            await this.initializeTimelineCache(this.restartEpoch);
        } catch (error) {
            if (error instanceof DOMException && error.name === "AbortError") {
                this.timelineStatus = "cache folder selection canceled";
                return;
            }

            const text = errorToString(error);
            this.timelineStatus = `cache folder error: ${text}`;
            this.onErr?.(text);
        }
    }

    async playTimeline() {
        if (this.timelineIsBusy || this.timelineIsPlaying) return;

        const runToken = ++this.timelineRunToken;
        let nextFrameTimeMs = performance.now() + TIMELINE_FRAME_INTERVAL_MS;
        this.timelineIsPlaying = true;

        try {
            while (
                runToken === this.timelineRunToken
                && this.timelineIsPlaying
                && this.timelineFrame < this.timelineFrameCount - 1
            ) {
                const nowMs = performance.now();
                if (nowMs < nextFrameTimeMs) {
                    const delayMs = nextFrameTimeMs - nowMs;
                    await sleep(delayMs);
                    continue;
                }

                await this.seekTimelineFrame(
                    this.timelineFrame + 1,
                    {
                        keepPlaying: true,
                        runToken,
                    },
                );

                if (runToken !== this.timelineRunToken || !this.timelineIsPlaying) {
                    return;
                }

                nextFrameTimeMs = Math.max(
                    nextFrameTimeMs + TIMELINE_FRAME_INTERVAL_MS,
                    performance.now() + TIMELINE_FRAME_INTERVAL_MS,
                );
            }
        } finally {
            if (runToken === this.timelineRunToken) {
                this.timelineIsPlaying = false;
            }
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
        this.requestStillFrameRender();
    }

    requestStillFrameRender() {
        if (!this.timeline) return;

        this.stillFrameRenderRequested = true;
        this.scheduleStillFrameRenderFlush();
    }

    private scheduleStillFrameRenderFlush() {
        if (this.stillFrameRenderInFlight) return;
        if (this.stillFrameRenderHandle !== null) return;

        if (typeof requestAnimationFrame === "undefined") {
            this.flushStillFrameRenderRequest();
            return;
        }

        this.stillFrameRenderHandle = requestAnimationFrame(() => {
            this.stillFrameRenderHandle = null;
            this.flushStillFrameRenderRequest();
        });
    }

    private flushStillFrameRenderRequest() {
        if (!this.stillFrameRenderRequested) return;
        if (this.stillFrameRenderInFlight) return;

        this.stillFrameRenderRequested = false;
        if (
            this.runner === null
            || !this.timeline
            || this.timelineIsBusy
            || this.timelineIsPlaying
        ) {
            return;
        }

        this.stillFrameRenderInFlight = true;
        void this.submitStillFrameRender();
    }

    private async submitStillFrameRender() {
        try {
            await this.runner?.renderStillFrame(this.frameTimingCallbacks);
            await this.device?.queue.onSubmittedWorkDone();
        } catch (error) {
            console.error(error);
            this.onErr?.(errorToString(error));
        } finally {
            this.stillFrameRenderInFlight = false;
            if (this.stillFrameRenderRequested) {
                this.scheduleStillFrameRenderFlush();
            }
        }
    }

    turnCamera(movement: { x: number, y: number }) {
        this.orbit.turn(movement);
        this.requestStillFrameRender();
    }

    panCamera(movement: { x: number, y: number }) {
        this.orbit.pan(movement);
        this.requestStillFrameRender();
    }

    zoomCamera(deltaY: number) {
        this.orbit.radius *= 2 ** (deltaY * 0.001);
        this.requestStillFrameRender();
    }


    async restart() {
        if (this.runner === null || this.device === null) return;

        const restartEpoch = ++this.restartEpoch;

        this.stopSimulation?.();
        this.stopSimulation = null;

        this.onStatusChange?.("initializing particles...");
        await waitForTimelineWorkYield();
        if (restartEpoch !== this.restartEpoch) return;
        
        try {
            this.runner.scatterParticles();

            await this.device.queue.onSubmittedWorkDone(); // need this to set simulation start time accurately

            if (restartEpoch !== this.restartEpoch) return;

            if (this.timeline) {
                await this.initializeTimelineCache(restartEpoch);
                return;
            }
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
        getScene = () => defaultSimulationScene,
        timeline = false,
        canvasPromise,
        onStatusChange,
        onErr,
    }: {
        getScene?: () => SimulationSceneConfig,
        timeline?: boolean,
        canvasPromise: Promise<HTMLCanvasElement>,
        onStatusChange?: (status: string) => void,
        onErr?: (err: string) => void,
    }) {
        const scene = getScene();
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
            timeline: timeline,
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
                    const { spawnSource, particleAppearances } = await loadSpawnSource(scene, state.nParticles);
                    if (destroyed) return;

                    updateStatus("loading collider...");
                    const collider = await loadCollider(scene);
                    if (destroyed) return;

                    updateStatus("loading environment...");
                    const environmentImageBitmap = await loadEnvironmentMap();
                    if (destroyed) return;

                    updateStatus("initializing renderer...");
                    await waitForTimelineWorkYield();
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
                        explicitMpmMaxSimulationTimestepS: () => state.explicitMpmMaxSimulationTimestepS,
                        mlsMpmMaxSimulationTimestepS: () => state.mlsMpmMaxSimulationTimestepS,
                        camera: state.camera,
                        spawnSource,
                        collider,
                        particleAppearances,
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
            state.pauseTimeline();
            if (state.stillFrameRenderHandle !== null) {
                cancelAnimationFrame(state.stillFrameRenderHandle);
                state.stillFrameRenderHandle = null;
            }
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
