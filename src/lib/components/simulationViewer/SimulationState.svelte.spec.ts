import { afterEach, describe, expect, it, vi } from "vitest";

import { SimulationState } from "./SimulationState.svelte";
import {
    TIMELINE_FRAME_COUNT,
    TIMELINE_FRAME_INTERVAL_MS,
} from "./SimulationTimelineTiming";
import {
    SIMULATION_FRAME_CACHE_DIRECTORY_NAME,
    simulationFrameFileName,
} from "./SimulationFrameCacheNames";

const TIMELINE_BACKGROUND_WORK_YIELD_MS = 250;

type MockFrameTimingCallbacks = {
    onGpuTimeUpdate?: (times: {
        computeSimulationStepNs: bigint,
        computeSimulationSubstepNs: bigint,
        nSimulationSubsteps: number,
        renderNs: bigint,
        postprocessRenderNs: bigint,
    }) => void,
    onAnimationFrameTimeUpdate?: (ms: number) => void,
};

type MockRunner = {
    renderStillFrame: (callbacks?: MockFrameTimingCallbacks) => void | Promise<void>,
};

type MockTimelineRunner = MockRunner & {
    selectedSimulationTimestepS: number,
    scatterParticles?: () => void,
    restoreSimulationPlaybackFrame?: (frame: ArrayBuffer) => void,
    advanceFixedSimulationSubsteps: (args: MockFrameTimingCallbacks & {
        nSubsteps: number,
    }) => {
        nSimulationSubsteps: number,
        simulationTimestepS: number,
        simulatedTimeS: number,
    },
    readSimulationPlaybackFrame: () => Promise<ArrayBuffer>,
};

type MockTimelineCache = {
    storageLabel: string,
    clear: () => Promise<void>,
    estimateFrameCapacity: (options: {
        frameByteLength: number,
        requestedFrameCount: number,
    }) => Promise<{
        frameCount: number,
        availableByteLength: number | null,
        quotaByteLength: number | null,
        usageByteLength: number | null,
    }>,
    readFrame: (frameIndex: number) => Promise<ArrayBuffer | null>,
    writeFrame: (frameIndex: number, snapshot: ArrayBuffer) => Promise<void>,
};

const attachRunner = (
    state: SimulationState,
    runner: MockRunner,
) => {
    (
        state as unknown as {
            runner: typeof runner,
        }
    ).runner = runner;
};

const attachTimelineCache = (
    state: SimulationState,
    timelineCache: MockTimelineCache,
) => {
    (
        state as unknown as {
            timelineCache: typeof timelineCache,
        }
    ).timelineCache = timelineCache;
};

const attachDevice = (
    state: SimulationState,
    device: {
        queue: {
            onSubmittedWorkDone: () => Promise<void>,
        },
    },
) => {
    (
        state as unknown as {
            device: typeof device,
        }
    ).device = device;
};

const attachCurrentTimelineCacheKey = (state: SimulationState) => {
    (
        state as unknown as {
            timelineCacheKey: string,
            currentTimelineCacheKey: () => string,
        }
    ).timelineCacheKey = (
        state as unknown as {
            currentTimelineCacheKey: () => string,
        }
    ).currentTimelineCacheKey();
};

const installAnimationFrameQueue = () => {
    const callbacks: FrameRequestCallback[] = [];

    vi.stubGlobal("requestAnimationFrame", vi.fn((callback: FrameRequestCallback) => {
        callbacks.push(callback);

        return callbacks.length;
    }));
    vi.stubGlobal("cancelAnimationFrame", vi.fn());

    return {
        flushNextFrame: async () => {
            const callback = callbacks.shift();
            expect(callback).toBeDefined();

            callback?.(0);
            await Promise.resolve();
            await Promise.resolve();
        },
        queuedFrameCount: () => callbacks.length,
    };
};

const installSuspendedAnimationFrame = () => {
    vi.stubGlobal("requestAnimationFrame", vi.fn(() => 1));
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
};

const flushMicrotasks = async () => {
    for (let i = 0; i < 5; i++) {
        await Promise.resolve();
    }
};

const estimateFullTimelineCapacity = vi.fn(async ({
    requestedFrameCount,
}: {
    requestedFrameCount: number,
}) => ({
    frameCount: requestedFrameCount,
    availableByteLength: null,
    quotaByteLength: null,
    usageByteLength: null,
}));

const createMockSelectedCacheDirectory = ({
    frameByteLength,
    cachedFrameCount = 0,
}: {
    frameByteLength: number,
    cachedFrameCount?: number,
}) => {
    const writable = {
        write: vi.fn(async () => {}),
        close: vi.fn(async () => {}),
        abort: vi.fn(async () => {}),
    };
    const frameDirectory = {
        entries: async function* () {
            for (let frameIndex = 0; frameIndex < cachedFrameCount; frameIndex++) {
                yield [simulationFrameFileName(frameIndex), {}] as [string, unknown];
            }
        },
        removeEntry: vi.fn(async () => {}),
        getFileHandle: vi.fn(async () => ({
            createWritable: vi.fn(async () => writable),
            getFile: vi.fn(async () => ({
                size: frameByteLength,
                arrayBuffer: vi.fn(async () => new ArrayBuffer(frameByteLength)),
            })),
        })),
    };
    const cacheRootDirectory = {
        entries: async function* () {},
        removeEntry: vi.fn(async () => {}),
        getDirectoryHandle: vi.fn(async () => frameDirectory),
    };
    const rootDirectory = {
        name: "sim-cache",
        entries: async function* () {},
        queryPermission: vi.fn(async () => "granted" as PermissionState),
        requestPermission: vi.fn(async () => "granted" as PermissionState),
        getDirectoryHandle: vi.fn(async (name: string) => {
            if (name === SIMULATION_FRAME_CACHE_DIRECTORY_NAME) {
                return cacheRootDirectory;
            }

            throw new DOMException("Directory not found", "NotFoundError");
        }),
    };

    return {
        cacheRootDirectory,
        frameDirectory,
        rootDirectory,
        writable,
    };
};

describe("SimulationState camera still-frame rendering", () => {
    afterEach(() => {
        vi.useRealTimers();
        vi.restoreAllMocks();
        vi.unstubAllGlobals();
    });

    it("coalesces timeline camera movement into one still-frame render", async () => {
        const animationFrames = installAnimationFrameQueue();
        const state = new SimulationState({ timeline: true });
        const runner = {
            renderStillFrame: vi.fn(() => {}),
        };

        attachRunner(state, runner);

        state.turnCamera({ x: 8, y: -4 });
        state.panCamera({ x: -2, y: 5 });
        state.zoomCamera(120);

        expect(animationFrames.queuedFrameCount()).toBe(1);
        expect(runner.renderStillFrame).not.toHaveBeenCalled();

        await animationFrames.flushNextFrame();

        expect(runner.renderStillFrame).toHaveBeenCalledTimes(1);
    });

    it("coalesces camera render requests while a still-frame render is in flight", async () => {
        const animationFrames = installAnimationFrameQueue();
        const state = new SimulationState({ timeline: true });
        let finishRender = () => {};
        const renderPromise = new Promise<void>((resolve) => {
            finishRender = resolve;
        });
        const runner = {
            renderStillFrame: vi.fn(() => renderPromise),
        };

        attachRunner(state, runner);

        state.turnCamera({ x: 8, y: -4 });
        await animationFrames.flushNextFrame();

        state.turnCamera({ x: -3, y: 6 });

        expect(animationFrames.queuedFrameCount()).toBe(0);
        expect(runner.renderStillFrame).toHaveBeenCalledTimes(1);

        finishRender();
        await Promise.resolve();
        await Promise.resolve();

        expect(animationFrames.queuedFrameCount()).toBe(1);

        await animationFrames.flushNextFrame();

        expect(runner.renderStillFrame).toHaveBeenCalledTimes(2);
    });

    it("keeps camera renders pending while timeline cache work is busy", async () => {
        const animationFrames = installAnimationFrameQueue();
        const state = new SimulationState({ timeline: true });
        const runner = {
            renderStillFrame: vi.fn(() => {}),
        };

        attachRunner(state, runner);

        state.timelineIsBusy = true;
        state.turnCamera({ x: 8, y: -4 });

        await animationFrames.flushNextFrame();

        expect(runner.renderStillFrame).not.toHaveBeenCalled();
        expect(animationFrames.queuedFrameCount()).toBe(0);

        (
            state as unknown as {
                releaseTimelineBusy: (runToken: number) => void,
            }
        ).releaseTimelineBusy(0);

        expect(animationFrames.queuedFrameCount()).toBe(1);

        await animationFrames.flushNextFrame();

        expect(runner.renderStillFrame).toHaveBeenCalledTimes(1);
    });

    it("does not wait for submitted GPU work before accepting the next camera render", async () => {
        const animationFrames = installAnimationFrameQueue();
        const state = new SimulationState({ timeline: true });
        const runner = {
            renderStillFrame: vi.fn(() => {}),
        };
        const device = {
            queue: {
                onSubmittedWorkDone: vi.fn(async () => {}),
            },
        };

        attachRunner(state, runner);
        attachDevice(state, device);

        state.turnCamera({ x: 8, y: -4 });
        await animationFrames.flushNextFrame();

        state.turnCamera({ x: -3, y: 6 });

        expect(animationFrames.queuedFrameCount()).toBe(1);

        await animationFrames.flushNextFrame();

        expect(runner.renderStillFrame).toHaveBeenCalledTimes(2);
        expect(device.queue.onSubmittedWorkDone).not.toHaveBeenCalled();
    });

    it("feeds timeline still-frame timings into the profiling state", async () => {
        const animationFrames = installAnimationFrameQueue();
        const state = new SimulationState({ timeline: true });
        const runner = {
            renderStillFrame: vi.fn((callbacks?: MockFrameTimingCallbacks) => {
                callbacks?.onAnimationFrameTimeUpdate?.(3.5);
                callbacks?.onGpuTimeUpdate?.({
                    computeSimulationStepNs: 0n,
                    computeSimulationSubstepNs: 0n,
                    nSimulationSubsteps: 0,
                    renderNs: 123n,
                    postprocessRenderNs: 45n,
                });
            }),
        };

        attachRunner(state, runner);

        state.zoomCamera(-120);
        await animationFrames.flushNextFrame();

        expect(state.elapsedTime.animationFrameTimeNs).toBe(3_500_000n);
        expect(state.elapsedTime.gpuComputeSimulationStepTimeNs).toBe(0n);
        expect(state.elapsedTime.gpuComputeSimulationSubstepTimeNs).toBe(0n);
        expect(state.elapsedTime.nSimulationSubsteps).toBe(0);
        expect(state.elapsedTime.gpuRenderTimeNs).toBe(123n);
        expect(state.elapsedTime.gpuPostprocessRenderTimeNs).toBe(45n);
    });

    it("does not schedule still-frame renders while the realtime loop is active", () => {
        const animationFrames = installAnimationFrameQueue();
        const state = new SimulationState({ timeline: false });
        const runner = {
            renderStillFrame: vi.fn(() => {}),
        };

        attachRunner(state, runner);

        state.turnCamera({ x: 8, y: -4 });
        state.panCamera({ x: -2, y: 5 });
        state.zoomCamera(120);

        expect(animationFrames.queuedFrameCount()).toBe(0);
        expect(runner.renderStillFrame).not.toHaveBeenCalled();
    });

    it("rerenders a timeline still frame after viewport resize", async () => {
        const animationFrames = installAnimationFrameQueue();
        const state = new SimulationState({ timeline: true });
        const runner = {
            renderStillFrame: vi.fn(() => {}),
        };

        attachRunner(state, runner);

        state.setViewportWidth(640);
        state.setViewportHeight(360);

        expect(state.width).toBe(640);
        expect(state.height).toBe(360);
        expect(animationFrames.queuedFrameCount()).toBe(1);

        await animationFrames.flushNextFrame();

        expect(runner.renderStillFrame).toHaveBeenCalledTimes(1);
    });

    it("initializes timeline caching when animation frames are suspended", async () => {
        vi.useFakeTimers();
        installSuspendedAnimationFrame();

        const state = new SimulationState({ timeline: true });
        const runner = {
            renderStillFrame: vi.fn(() => {}),
            simulationPlaybackFrameLayout: {
                byteLength: 8,
            },
            readSimulationPlaybackFrame: vi.fn(async () => new ArrayBuffer(8)),
        };
        const timelineCache: MockTimelineCache = {
            storageLabel: "test file cache",
            clear: vi.fn(async () => {}),
            estimateFrameCapacity: estimateFullTimelineCapacity,
            readFrame: vi.fn(async () => null),
            writeFrame: vi.fn(async () => {}),
        };

        attachRunner(state, runner);
        attachTimelineCache(state, timelineCache);
        attachCurrentTimelineCacheKey(state);

        const initializePromise = (
            state as unknown as {
                initializeTimelineCache: (restartEpoch: number) => Promise<void>,
            }
        ).initializeTimelineCache(0);
        await Promise.resolve();

        expect(runner.renderStillFrame).not.toHaveBeenCalled();
        expect(timelineCache.writeFrame).not.toHaveBeenCalled();

        await vi.advanceTimersByTimeAsync(TIMELINE_BACKGROUND_WORK_YIELD_MS);
        await initializePromise;

        expect(runner.renderStillFrame).toHaveBeenCalledTimes(1);
        expect(timelineCache.writeFrame).toHaveBeenCalledTimes(1);
        expect(state.timelineNextUncachedFrame).toBe(1);
        expect(state.timelineStatus).toContain("cached frame 0");
    });

    it("waits for browser paint before foreground timeline caching", async () => {
        vi.useFakeTimers();
        const animationFrames = installAnimationFrameQueue();

        const state = new SimulationState({ timeline: true });
        const runner = {
            renderStillFrame: vi.fn(() => {}),
            simulationPlaybackFrameLayout: {
                byteLength: 8,
            },
            readSimulationPlaybackFrame: vi.fn(async () => new ArrayBuffer(8)),
        };
        const timelineCache: MockTimelineCache = {
            storageLabel: "test file cache",
            clear: vi.fn(async () => {}),
            estimateFrameCapacity: estimateFullTimelineCapacity,
            readFrame: vi.fn(async () => null),
            writeFrame: vi.fn(async () => {}),
        };

        attachRunner(state, runner);
        attachTimelineCache(state, timelineCache);
        attachCurrentTimelineCacheKey(state);

        const initializePromise = (
            state as unknown as {
                initializeTimelineCache: (restartEpoch: number) => Promise<void>,
            }
        ).initializeTimelineCache(0);
        await flushMicrotasks();

        expect(animationFrames.queuedFrameCount()).toBe(1);
        expect(runner.renderStillFrame).not.toHaveBeenCalled();
        expect(timelineCache.writeFrame).not.toHaveBeenCalled();

        await vi.advanceTimersByTimeAsync(16);

        expect(runner.renderStillFrame).not.toHaveBeenCalled();
        expect(timelineCache.writeFrame).not.toHaveBeenCalled();

        await animationFrames.flushNextFrame();

        expect(animationFrames.queuedFrameCount()).toBe(1);
        expect(runner.renderStillFrame).not.toHaveBeenCalled();
        expect(timelineCache.writeFrame).not.toHaveBeenCalled();

        await animationFrames.flushNextFrame();
        await initializePromise;

        expect(runner.renderStillFrame).toHaveBeenCalledTimes(1);
        expect(timelineCache.writeFrame).toHaveBeenCalledTimes(1);
        expect(state.timelineNextUncachedFrame).toBe(1);
    });

    it("continues baking timeline frames when animation frames are suspended", async () => {
        vi.useFakeTimers();
        installSuspendedAnimationFrame();

        const simulationTimestepS = 0.01;
        const state = new SimulationState({ timeline: true });
        const runner: MockTimelineRunner = {
            selectedSimulationTimestepS: simulationTimestepS,
            renderStillFrame: vi.fn(() => {}),
            advanceFixedSimulationSubsteps: vi.fn(({
                nSubsteps,
            }) => ({
                nSimulationSubsteps: nSubsteps,
                simulationTimestepS,
                simulatedTimeS: nSubsteps * simulationTimestepS,
            })),
            readSimulationPlaybackFrame: vi.fn(async () => new ArrayBuffer(8)),
        };
        const timelineCache: MockTimelineCache = {
            storageLabel: "test file cache",
            clear: vi.fn(async () => {}),
            estimateFrameCapacity: estimateFullTimelineCapacity,
            readFrame: vi.fn(async () => null),
            writeFrame: vi.fn(async () => {}),
        };

        attachRunner(state, runner);
        attachTimelineCache(state, timelineCache);
        state.timelineNextUncachedFrame = 1;
        state.timelineFrame = 0;

        const bakePromise = state.setTimelineFrame(2);
        await vi.advanceTimersByTimeAsync(TIMELINE_BACKGROUND_WORK_YIELD_MS);
        await vi.advanceTimersByTimeAsync(TIMELINE_BACKGROUND_WORK_YIELD_MS);
        await bakePromise;

        expect(runner.advanceFixedSimulationSubsteps).toHaveBeenCalledTimes(2);
        expect(timelineCache.writeFrame).toHaveBeenCalledTimes(2);
        expect(state.timelineFrame).toBe(2);
        expect(state.timelineNextUncachedFrame).toBe(3);
    });

    it("bakes timeline frames by simulated seconds instead of one solver max step per frame", async () => {
        const simulationTimestepS = 1 / 1024;
        const advancedSubsteps: number[] = [];
        const state = new SimulationState({ timeline: true });
        const runner: MockTimelineRunner = {
            selectedSimulationTimestepS: simulationTimestepS,
            renderStillFrame: vi.fn(() => {}),
            advanceFixedSimulationSubsteps: vi.fn(({
                nSubsteps,
            }) => {
                advancedSubsteps.push(nSubsteps);

                return {
                    nSimulationSubsteps: nSubsteps,
                    simulationTimestepS,
                    simulatedTimeS: nSubsteps * simulationTimestepS,
                };
            }),
            readSimulationPlaybackFrame: vi.fn(async () => new ArrayBuffer(8)),
        };
        const timelineCache: MockTimelineCache = {
            storageLabel: "test file cache",
            clear: vi.fn(async () => {}),
            estimateFrameCapacity: estimateFullTimelineCapacity,
            readFrame: vi.fn(async () => null),
            writeFrame: vi.fn(async () => {}),
        };

        attachRunner(state, runner);
        attachTimelineCache(state, timelineCache);
        state.timelineNextUncachedFrame = 1;
        state.timelineFrame = 0;

        await state.setTimelineFrame(30);

        expect(advancedSubsteps.reduce((total, nSubsteps) => total + nSubsteps, 0)).toBe(1024);
        expect(runner.advanceFixedSimulationSubsteps).toHaveBeenCalledTimes(30);
        expect(timelineCache.writeFrame).toHaveBeenCalledTimes(30);
        expect(state.timelineFrame).toBe(30);
        expect(state.timelineNextUncachedFrame).toBe(31);
    });

    it("uses the selected timeline step divisor when baking frames", async () => {
        const simulationTimestepS = 0.01;
        const advancedSubsteps: number[] = [];
        const state = new SimulationState({ timeline: true });
        await state.setTimelineStepDivisor(10);
        const runner: MockTimelineRunner = {
            selectedSimulationTimestepS: simulationTimestepS,
            renderStillFrame: vi.fn(() => {}),
            advanceFixedSimulationSubsteps: vi.fn(({
                nSubsteps,
            }) => {
                advancedSubsteps.push(nSubsteps);

                return {
                    nSimulationSubsteps: nSubsteps,
                    simulationTimestepS,
                    simulatedTimeS: nSubsteps * simulationTimestepS,
                };
            }),
            readSimulationPlaybackFrame: vi.fn(async () => new ArrayBuffer(8)),
        };
        const timelineCache: MockTimelineCache = {
            storageLabel: "test file cache",
            clear: vi.fn(async () => {}),
            estimateFrameCapacity: estimateFullTimelineCapacity,
            readFrame: vi.fn(async () => null),
            writeFrame: vi.fn(async () => {}),
        };

        attachRunner(state, runner);
        attachTimelineCache(state, timelineCache);
        state.timelineNextUncachedFrame = 1;
        state.timelineFrame = 0;

        await state.setTimelineFrame(2);

        expect(advancedSubsteps).toEqual([10, 10]);
        expect(state.timelineStepDivisor).toBe(10);
        expect(state.timelineSecondsPerFrame).toBe(0.1);
        expect(state.timelineFrame).toBe(2);
        expect(state.timelineNextUncachedFrame).toBe(3);
    });

    it("clamps selected timeline step divisors into the supported range", async () => {
        const state = new SimulationState({ timeline: true });

        await state.setTimelineStepDivisor(0);
        expect(state.timelineSecondsPerFrame).toBe(1);
        expect(state.timelineStepDivisor).toBe(1);

        await state.setTimelineStepDivisor(10_000);
        expect(state.timelineSecondsPerFrame).toBeCloseTo(1 / 240);
        expect(state.timelineStepDivisor).toBeCloseTo(240);
    });

    it("plays cached timeline frames sequentially instead of skipping to wall clock", async () => {
        vi.useFakeTimers();

        let nowMs = 0;
        vi.spyOn(performance, "now").mockImplementation(() => nowMs);

        const state = new SimulationState({ timeline: true });
        const runner = {
            renderStillFrame: vi.fn(() => {}),
            restoreSimulationPlaybackFrame: vi.fn(() => {}),
        };
        const timelineCache: MockTimelineCache = {
            storageLabel: "test file cache",
            clear: vi.fn(async () => {}),
            estimateFrameCapacity: estimateFullTimelineCapacity,
            readFrame: vi.fn(async () => new ArrayBuffer(8)),
            writeFrame: vi.fn(async () => {}),
        };

        attachRunner(state, runner);
        attachTimelineCache(state, timelineCache);
        state.timelineNextUncachedFrame = 91;
        state.timelineFrameCount = 2;
        state.timelineFrame = 0;

        const playPromise = state.playTimeline();
        await Promise.resolve();

        expect(timelineCache.readFrame).not.toHaveBeenCalled();

        nowMs = TIMELINE_FRAME_INTERVAL_MS * 5;
        await vi.advanceTimersByTimeAsync(TIMELINE_FRAME_INTERVAL_MS);
        await playPromise;

        expect(timelineCache.readFrame).toHaveBeenCalledWith(1);
        expect(timelineCache.readFrame).not.toHaveBeenCalledWith(5);
        expect(state.timelineFrame).toBe(1);
    });

    it("clamps the timeline to cached frames when file-cache quota is reached", async () => {
        const quotaError = new DOMException("Quota exceeded", "QuotaExceededError");
        const onErr = vi.fn();
        const onStatusChange = vi.fn();
        const state = new SimulationState({
            timeline: true,
            onErr,
            onStatusChange,
        });
        const runner = {
            selectedSimulationTimestepS: 1 / 1024,
            renderStillFrame: vi.fn(() => {}),
            restoreSimulationPlaybackFrame: vi.fn(() => {}),
            advanceFixedSimulationSubsteps: vi.fn(({
                nSubsteps,
            }: {
                nSubsteps: number,
            }) => ({
                nSimulationSubsteps: nSubsteps,
                simulationTimestepS: 1 / 1024,
                simulatedTimeS: nSubsteps / 1024,
            })),
            readSimulationPlaybackFrame: vi.fn(async () => new ArrayBuffer(8)),
        };
        const timelineCache: MockTimelineCache = {
            storageLabel: "test file cache",
            clear: vi.fn(async () => {}),
            estimateFrameCapacity: estimateFullTimelineCapacity,
            readFrame: vi.fn(async () => new ArrayBuffer(8)),
            writeFrame: vi.fn(async () => {
                throw quotaError;
            }),
        };

        attachRunner(state, runner);
        attachTimelineCache(state, timelineCache);
        state.timelineNextUncachedFrame = 1;
        state.timelineFrame = 0;

        await state.setTimelineFrame(2);

        expect(onErr).not.toHaveBeenCalled();
        expect(onStatusChange).toHaveBeenCalledWith("timeline cache full");
        expect(runner.restoreSimulationPlaybackFrame).toHaveBeenCalledTimes(1);
        expect(state.timelineFrame).toBe(0);
        expect(state.timelineFrameCount).toBe(1);
        expect(state.timelineIsBusy).toBe(false);
        expect(state.timelineStatus).toContain("cache quota reached");
    });

    it("keeps the full timeline range when the browser capacity estimate is below one frame", async () => {
        const state = new SimulationState({ timeline: true });
        const runner = {
            renderStillFrame: vi.fn(() => {}),
            simulationPlaybackFrameLayout: {
                byteLength: 4.6 * 1024 * 1024,
            },
        };
        const timelineCache: MockTimelineCache = {
            storageLabel: "test file cache",
            clear: vi.fn(async () => {}),
            estimateFrameCapacity: vi.fn(async () => ({
                frameCount: 0,
                availableByteLength: 4 * 1024 * 1024,
                quotaByteLength: 8 * 1024 * 1024,
                usageByteLength: 4 * 1024 * 1024,
            })),
            readFrame: vi.fn(async () => null),
            writeFrame: vi.fn(async () => {}),
        };

        attachRunner(state, runner);
        attachTimelineCache(state, timelineCache);
        (
            state as unknown as {
                timelineCacheKey: string,
                currentTimelineCacheKey: () => string,
            }
        ).timelineCacheKey = (
            state as unknown as {
                currentTimelineCacheKey: () => string,
            }
        ).currentTimelineCacheKey();

        const cacheHasRoom = await (
            state as unknown as {
                resetTimelineCache: () => Promise<boolean>,
            }
        ).resetTimelineCache();

        expect(cacheHasRoom).toBe(true);
        expect(state.timelineFrameCount).toBe(TIMELINE_FRAME_COUNT);
        expect(state.timelineStorageLabel).toContain("capacity estimate low");
        expect(state.timelineStatus).toContain("caching until a write fails");
    });

    it("rebuilds timeline cache in a selected folder without OPFS capacity capping", async () => {
        const frameByteLength = 8;
        const { rootDirectory, writable } = createMockSelectedCacheDirectory({
            frameByteLength,
        });
        const showDirectoryPicker = vi.fn(async () => rootDirectory);

        vi.stubGlobal("showDirectoryPicker", showDirectoryPicker);

        const state = new SimulationState({ timeline: true });
        const runner = {
            renderStillFrame: vi.fn(() => {}),
            simulationPlaybackFrameLayout: {
                byteLength: frameByteLength,
            },
            readSimulationPlaybackFrame: vi.fn(async () => new ArrayBuffer(frameByteLength)),
        };

        attachRunner(state, runner);

        await state.selectTimelineCacheDirectory();

        expect(state.timelineCanSelectCacheDirectory).toBe(true);
        expect(showDirectoryPicker).toHaveBeenCalledWith({
            id: "websnow-simulation-frame-cache",
            mode: "readwrite",
        });
        expect(rootDirectory.getDirectoryHandle).toHaveBeenCalledWith(
            "websnow-simulation-frame-cache",
            { create: true },
        );
        expect(writable.write).toHaveBeenCalledTimes(1);
        expect(writable.close).toHaveBeenCalledTimes(1);
        expect(state.timelineFrameCount).toBe(TIMELINE_FRAME_COUNT);
        expect(state.timelineNextUncachedFrame).toBe(1);
        expect(state.timelineStorageLabel).toBe("folder cache: sim-cache");
        expect(state.timelineStatus).toContain("cached frame 0");
    });

    it("loads existing frames from a selected cache folder instead of rebuilding them", async () => {
        vi.useFakeTimers();
        installSuspendedAnimationFrame();

        const frameByteLength = 8;
        const {
            frameDirectory,
            rootDirectory,
            writable,
        } = createMockSelectedCacheDirectory({
            frameByteLength,
            cachedFrameCount: 3,
        });
        const showDirectoryPicker = vi.fn(async () => rootDirectory);

        vi.stubGlobal("showDirectoryPicker", showDirectoryPicker);

        const state = new SimulationState({ timeline: true });
        const runner = {
            renderStillFrame: vi.fn(() => {}),
            restoreSimulationPlaybackFrame: vi.fn(() => {}),
            simulationPlaybackFrameLayout: {
                byteLength: frameByteLength,
            },
            readSimulationPlaybackFrame: vi.fn(async () => new ArrayBuffer(frameByteLength)),
        };

        attachRunner(state, runner);

        const selectPromise = state.selectTimelineCacheDirectory();
        await vi.advanceTimersByTimeAsync(TIMELINE_BACKGROUND_WORK_YIELD_MS);
        await selectPromise;

        expect(frameDirectory.removeEntry).not.toHaveBeenCalled();
        expect(writable.write).not.toHaveBeenCalled();
        expect(runner.readSimulationPlaybackFrame).not.toHaveBeenCalled();
        expect(runner.restoreSimulationPlaybackFrame).toHaveBeenCalledTimes(1);
        expect(state.timelineFrame).toBe(0);
        expect(state.timelineNextUncachedFrame).toBe(3);
        expect(state.timelineStatus).toContain("loaded 3 cached frames");
    });

    it("reports unsupported folder cache selection when the browser lacks a directory picker", async () => {
        vi.stubGlobal("showDirectoryPicker", undefined);

        const state = new SimulationState({ timeline: true });

        await state.selectTimelineCacheDirectory();

        expect(state.timelineCanSelectCacheDirectory).toBe(false);
        expect(state.timelineStatus).toContain("directory picker support");
    });

    it("play bakes past frame 0 when storage capacity was only an estimate", async () => {
        vi.useFakeTimers();

        let nowMs = 0;
        vi.spyOn(performance, "now").mockImplementation(() => nowMs);

        const state = new SimulationState({ timeline: true });
        const runner: MockTimelineRunner = {
            selectedSimulationTimestepS: 1 / 1024,
            renderStillFrame: vi.fn(() => {}),
            advanceFixedSimulationSubsteps: vi.fn(({
                nSubsteps,
            }) => ({
                nSimulationSubsteps: nSubsteps,
                simulationTimestepS: 1 / 1024,
                simulatedTimeS: nSubsteps / 1024,
            })),
            readSimulationPlaybackFrame: vi.fn(async () => new ArrayBuffer(8)),
        };
        const timelineCache: MockTimelineCache = {
            storageLabel: "test file cache",
            clear: vi.fn(async () => {}),
            estimateFrameCapacity: estimateFullTimelineCapacity,
            readFrame: vi.fn(async () => null),
            writeFrame: vi.fn(async () => {}),
        };

        attachRunner(state, runner);
        attachTimelineCache(state, timelineCache);
        state.timelineFrameCount = 2;
        state.timelineNextUncachedFrame = 1;
        state.timelineFrame = 0;

        const playPromise = state.playTimeline();
        await Promise.resolve();

        expect(timelineCache.writeFrame).not.toHaveBeenCalled();

        nowMs = TIMELINE_FRAME_INTERVAL_MS * 2;
        await vi.advanceTimersByTimeAsync(TIMELINE_FRAME_INTERVAL_MS * 2);
        await playPromise;

        expect(runner.advanceFixedSimulationSubsteps).toHaveBeenCalledTimes(1);
        expect(timelineCache.writeFrame).toHaveBeenCalledTimes(1);
        expect(state.timelineFrame).toBe(1);
    });

    it("finishes the active playback write after pause and releases timeline controls", async () => {
        vi.useFakeTimers();

        let nowMs = 0;
        vi.spyOn(performance, "now").mockImplementation(() => nowMs);

        let finishWrite = () => {};
        const writePromise = new Promise<void>((resolve) => {
            finishWrite = resolve;
        });

        const state = new SimulationState({ timeline: true });
        const runner: MockTimelineRunner = {
            selectedSimulationTimestepS: 1 / 1024,
            renderStillFrame: vi.fn(() => {}),
            advanceFixedSimulationSubsteps: vi.fn(({
                nSubsteps,
            }) => ({
                nSimulationSubsteps: nSubsteps,
                simulationTimestepS: 1 / 1024,
                simulatedTimeS: nSubsteps / 1024,
            })),
            readSimulationPlaybackFrame: vi.fn(async () => new ArrayBuffer(8)),
        };
        const timelineCache: MockTimelineCache = {
            storageLabel: "test file cache",
            clear: vi.fn(async () => {}),
            estimateFrameCapacity: estimateFullTimelineCapacity,
            readFrame: vi.fn(async () => null),
            writeFrame: vi.fn(() => writePromise),
        };

        attachRunner(state, runner);
        attachTimelineCache(state, timelineCache);
        state.timelineFrameCount = 2;
        state.timelineNextUncachedFrame = 1;
        state.timelineFrame = 0;

        const playPromise = state.playTimeline();
        await Promise.resolve();

        nowMs = TIMELINE_FRAME_INTERVAL_MS;
        await vi.advanceTimersByTimeAsync(TIMELINE_FRAME_INTERVAL_MS);
        await flushMicrotasks();

        expect(timelineCache.writeFrame).toHaveBeenCalledTimes(1);
        expect(state.timelineIsBusy).toBe(true);

        state.pauseTimeline();
        finishWrite();
        await playPromise;

        expect(state.timelineIsPlaying).toBe(false);
        expect(state.timelineIsBusy).toBe(false);
        expect(state.timelineFrame).toBe(1);
        expect(state.timelineNextUncachedFrame).toBe(2);
    });

    it("rebuilds solver state from scatter before baking past compact cached playback frames", async () => {
        const state = new SimulationState({ timeline: true });
        const runner: MockTimelineRunner = {
            selectedSimulationTimestepS: 1 / 1024,
            scatterParticles: vi.fn(() => {}),
            renderStillFrame: vi.fn(() => {}),
            restoreSimulationPlaybackFrame: vi.fn(() => {}),
            advanceFixedSimulationSubsteps: vi.fn(({
                nSubsteps,
            }) => ({
                nSimulationSubsteps: nSubsteps,
                simulationTimestepS: 1 / 1024,
                simulatedTimeS: nSubsteps / 1024,
            })),
            readSimulationPlaybackFrame: vi.fn(async () => new ArrayBuffer(8)),
        };
        const timelineCache: MockTimelineCache = {
            storageLabel: "test file cache",
            clear: vi.fn(async () => {}),
            estimateFrameCapacity: estimateFullTimelineCapacity,
            readFrame: vi.fn(async () => new ArrayBuffer(8)),
            writeFrame: vi.fn(async () => {}),
        };

        attachRunner(state, runner);
        attachTimelineCache(state, timelineCache);
        state.timelineNextUncachedFrame = 3;
        state.timelineFrame = 0;

        await state.setTimelineFrame(1);
        await state.setTimelineFrame(3);

        expect(runner.scatterParticles).toHaveBeenCalledTimes(1);
        expect(runner.advanceFixedSimulationSubsteps).toHaveBeenCalledTimes(3);
        expect(timelineCache.writeFrame).toHaveBeenCalledTimes(1);
        expect(state.timelineFrame).toBe(3);
        expect(state.timelineNextUncachedFrame).toBe(4);
    });
});
