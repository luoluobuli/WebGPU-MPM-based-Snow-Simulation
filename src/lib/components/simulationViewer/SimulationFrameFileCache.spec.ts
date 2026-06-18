import { afterEach, describe, expect, it, vi } from "vitest";

import {
    canSelectSimulationFrameCacheDirectory,
    createSimulationFrameFileCache,
    selectSimulationFrameCacheDirectory,
} from "./SimulationFrameFileCache";
import {
    SIMULATION_FRAME_CACHE_DIRECTORY_NAME,
    simulationFrameFileName,
} from "./SimulationFrameCacheNames";

type MockWritable = {
    write: ReturnType<typeof vi.fn>,
    close: ReturnType<typeof vi.fn>,
    abort: ReturnType<typeof vi.fn>,
};

const createMockOpfs = (
    writable: MockWritable,
    {
        cacheRootEntries = [],
        frameDirectoryEntries = [],
        estimate = async () => ({
            quota: 512 * 1024 * 1024,
            usage: 0,
        }),
    }: {
        cacheRootEntries?: string[],
        frameDirectoryEntries?: Array<{
            name: string,
            byteLength: number,
        }>,
        estimate?: () => Promise<StorageEstimate>,
    } = {},
) => {
    const frameDirectory = {
        entries: async function* () {
            for (const entry of frameDirectoryEntries) {
                yield [entry.name, {}] as [string, unknown];
            }
        },
        removeEntry: vi.fn(async () => {}),
        getFileHandle: vi.fn(async (name: string) => {
            const entry = frameDirectoryEntries.find((candidate) => candidate.name === name);

            return {
                createWritable: vi.fn(async () => writable),
                getFile: vi.fn(async () => ({
                    size: entry?.byteLength ?? 0,
                    arrayBuffer: vi.fn(async () => new ArrayBuffer(entry?.byteLength ?? 0)),
                })),
            };
        }),
    };
    const cacheRootDirectory = {
        entries: async function* () {
            for (const name of cacheRootEntries) {
                yield [name, {}] as [string, unknown];
            }
        },
        removeEntry: vi.fn(async () => {}),
        getDirectoryHandle: vi.fn(async () => frameDirectory),
    };
    const rootDirectory = {
        entries: async function* () {},
        getDirectoryHandle: vi.fn(async (name: string) => {
            if (name === SIMULATION_FRAME_CACHE_DIRECTORY_NAME) {
                return cacheRootDirectory;
            }

            throw new DOMException("Directory not found", "NotFoundError");
        }),
    };

    vi.stubGlobal("navigator", {
        storage: {
            getDirectory: vi.fn(async () => rootDirectory),
            estimate: vi.fn(estimate),
        },
    });

    return {
        cacheRootDirectory,
        frameDirectory,
    };
};

const createMockSelectedDirectory = (
    writable: MockWritable,
    {
        cacheRootEntries = [],
        rootEntries = [],
        frameDirectoryEntries = [],
    }: {
        cacheRootEntries?: string[],
        rootEntries?: string[],
        frameDirectoryEntries?: Array<{
            name: string,
            byteLength: number,
        }>,
    } = {},
) => {
    const frameDirectory = {
        entries: async function* () {
            for (const entry of frameDirectoryEntries) {
                yield [entry.name, {}] as [string, unknown];
            }
        },
        removeEntry: vi.fn(async () => {}),
        getFileHandle: vi.fn(async (name: string) => {
            const entry = frameDirectoryEntries.find((candidate) => candidate.name === name);

            return {
                createWritable: vi.fn(async () => writable),
                getFile: vi.fn(async () => ({
                    size: entry?.byteLength ?? 0,
                    arrayBuffer: vi.fn(async () => new ArrayBuffer(entry?.byteLength ?? 0)),
                })),
            };
        }),
    };
    const cacheRootDirectory = {
        entries: async function* () {
            for (const name of cacheRootEntries) {
                yield [name, {}] as [string, unknown];
            }
        },
        removeEntry: vi.fn(async () => {}),
        getDirectoryHandle: vi.fn(async () => frameDirectory),
    };
    const rootDirectory = {
        name: "sim-cache",
        entries: async function* () {
            for (const name of rootEntries) {
                yield [name, {}] as [string, unknown];
            }
        },
        removeEntry: frameDirectory.removeEntry,
        getFileHandle: frameDirectory.getFileHandle,
        getDirectoryHandle: vi.fn(async (name: string) => {
            if (name === SIMULATION_FRAME_CACHE_DIRECTORY_NAME) {
                return cacheRootDirectory;
            }

            throw new DOMException("Directory not found", "NotFoundError");
        }),
    } as unknown as FileSystemDirectoryHandle;

    return {
        cacheRootDirectory,
        frameDirectory,
        rootDirectory,
    };
};

describe("SimulationFrameFileCache", () => {
    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it("does not close an OPFS stream after a failed frame write", async () => {
        const writeError = new Error("quota exceeded");
        const writable = {
            write: vi.fn(async () => {
                throw writeError;
            }),
            close: vi.fn(async () => {}),
            abort: vi.fn(async () => {}),
        };
        const { frameDirectory } = createMockOpfs(writable);
        const cache = await createSimulationFrameFileCache({ cacheKey: "failing-frame" });

        await expect(cache.writeFrame(1, new ArrayBuffer(16))).rejects.toThrow(writeError);

        expect(writable.close).not.toHaveBeenCalled();
        expect(writable.abort).toHaveBeenCalledTimes(1);
        expect(frameDirectory.removeEntry).toHaveBeenCalledWith("frame-000001.bin");
    });

    it("chunks large OPFS frame writes before closing the stream", async () => {
        const writable = {
            write: vi.fn(async () => {}),
            close: vi.fn(async () => {}),
            abort: vi.fn(async () => {}),
        };
        createMockOpfs(writable);
        const cache = await createSimulationFrameFileCache({ cacheKey: "large-frame" });

        await cache.writeFrame(2, new ArrayBuffer(17 * 1024 * 1024));

        expect(writable.write).toHaveBeenCalledTimes(3);
        expect(writable.close).toHaveBeenCalledTimes(1);
        expect(writable.abort).not.toHaveBeenCalled();
    });

    it("removes stale app cache directories before opening the active cache key", async () => {
        const writable = {
            write: vi.fn(async () => {}),
            close: vi.fn(async () => {}),
            abort: vi.fn(async () => {}),
        };
        const { cacheRootDirectory } = createMockOpfs(
            writable,
            {
                cacheRootEntries: [
                    "old-cache",
                    "active-cache",
                ],
            },
        );

        await createSimulationFrameFileCache({ cacheKey: "active-cache" });

        expect(cacheRootDirectory.removeEntry).toHaveBeenCalledWith(
            "old-cache",
            { recursive: true },
        );
        expect(cacheRootDirectory.removeEntry).not.toHaveBeenCalledWith(
            "active-cache",
            { recursive: true },
        );
    });

    it("uses selected folders as disk caches without OPFS quota estimates", async () => {
        const writable = {
            write: vi.fn(async () => {}),
            close: vi.fn(async () => {}),
            abort: vi.fn(async () => {}),
        };
        const { cacheRootDirectory, rootDirectory } = createMockSelectedDirectory(
            writable,
            {
                cacheRootEntries: [
                    "old-cache",
                    "selected-cache",
                ],
            },
        );

        const cache = await createSimulationFrameFileCache({
            cacheKey: "selected-cache",
            rootDirectory,
        });

        await expect(cache.estimateFrameCapacity({
            frameByteLength: 100 * 1024 * 1024,
            requestedFrameCount: 180,
        })).resolves.toMatchObject({
            frameCount: 180,
            availableByteLength: null,
        });

        expect(cache.storageLabel).toBe("folder cache: sim-cache");
        expect(rootDirectory.getDirectoryHandle).toHaveBeenCalledWith(
            "websnow-simulation-frame-cache",
            { create: true },
        );
        expect(cacheRootDirectory.removeEntry).toHaveBeenCalledWith(
            "old-cache",
            { recursive: true },
        );
        expect(cacheRootDirectory.removeEntry).not.toHaveBeenCalledWith(
            "selected-cache",
            { recursive: true },
        );
    });

    it("finds the next uncached frame in a selected folder cache", async () => {
        const writable = {
            write: vi.fn(async () => {}),
            close: vi.fn(async () => {}),
            abort: vi.fn(async () => {}),
        };
        const { cacheRootDirectory, frameDirectory, rootDirectory } = createMockSelectedDirectory(
            writable,
            {
                frameDirectoryEntries: [
                    {
                        name: simulationFrameFileName(0),
                        byteLength: 16,
                    },
                    {
                        name: simulationFrameFileName(1),
                        byteLength: 16,
                    },
                    {
                        name: simulationFrameFileName(3),
                        byteLength: 16,
                    },
                ],
            },
        );

        const cache = await createSimulationFrameFileCache({
            cacheKey: "selected-cache",
            rootDirectory,
        });

        await expect(cache.findNextUncachedFrame?.({
            frameByteLength: 16,
            requestedFrameCount: 180,
        })).resolves.toBe(2);
        expect(frameDirectory.removeEntry).not.toHaveBeenCalled();
        expect(cacheRootDirectory.removeEntry).not.toHaveBeenCalledWith(
            "selected-cache",
            { recursive: true },
        );
    });

    it("does not adopt selected folder cache frames with the wrong byte length", async () => {
        const writable = {
            write: vi.fn(async () => {}),
            close: vi.fn(async () => {}),
            abort: vi.fn(async () => {}),
        };
        const { rootDirectory } = createMockSelectedDirectory(
            writable,
            {
                frameDirectoryEntries: [
                    {
                        name: simulationFrameFileName(0),
                        byteLength: 12,
                    },
                    {
                        name: simulationFrameFileName(1),
                        byteLength: 16,
                    },
                ],
            },
        );

        const cache = await createSimulationFrameFileCache({
            cacheKey: "selected-cache",
            rootDirectory,
        });

        await expect(cache.findNextUncachedFrame?.({
            frameByteLength: 16,
            requestedFrameCount: 180,
        })).resolves.toBe(0);
    });

    it("can use a selected folder that already contains cache frame files", async () => {
        const writable = {
            write: vi.fn(async () => {}),
            close: vi.fn(async () => {}),
            abort: vi.fn(async () => {}),
        };
        const { cacheRootDirectory, rootDirectory } = createMockSelectedDirectory(
            writable,
            {
                rootEntries: [
                    simulationFrameFileName(0),
                    simulationFrameFileName(1),
                ],
                frameDirectoryEntries: [
                    {
                        name: simulationFrameFileName(0),
                        byteLength: 16,
                    },
                    {
                        name: simulationFrameFileName(1),
                        byteLength: 16,
                    },
                ],
            },
        );

        const cache = await createSimulationFrameFileCache({
            cacheKey: "selected-cache",
            rootDirectory,
        });

        await expect(cache.findNextUncachedFrame?.({
            frameByteLength: 16,
            requestedFrameCount: 180,
        })).resolves.toBe(2);
        expect(rootDirectory.getDirectoryHandle).not.toHaveBeenCalledWith(
            SIMULATION_FRAME_CACHE_DIRECTORY_NAME,
            { create: true },
        );
        expect(cacheRootDirectory.removeEntry).not.toHaveBeenCalled();
    });

    it("selects a writable cache folder with the browser directory picker", async () => {
        const directory = {
            name: "picked-cache",
            queryPermission: vi.fn(async () => "prompt" as PermissionState),
            requestPermission: vi.fn(async () => "granted" as PermissionState),
        };
        const showDirectoryPicker = vi.fn(async () => directory);

        vi.stubGlobal("showDirectoryPicker", showDirectoryPicker);

        expect(canSelectSimulationFrameCacheDirectory()).toBe(true);
        await expect(selectSimulationFrameCacheDirectory()).resolves.toBe(directory);
        expect(showDirectoryPicker).toHaveBeenCalledWith({
            id: "websnow-simulation-frame-cache",
            mode: "readwrite",
        });
        expect(directory.queryPermission).toHaveBeenCalledWith({ mode: "readwrite" });
        expect(directory.requestPermission).toHaveBeenCalledWith({ mode: "readwrite" });
    });

    it("estimates OPFS frame capacity with a safety margin", async () => {
        const writable = {
            write: vi.fn(async () => {}),
            close: vi.fn(async () => {}),
            abort: vi.fn(async () => {}),
        };
        createMockOpfs(
            writable,
            {
                estimate: async () => ({
                    quota: 512 * 1024 * 1024,
                    usage: 0,
                }),
            },
        );
        const cache = await createSimulationFrameFileCache({ cacheKey: "capacity" });

        await expect(cache.estimateFrameCapacity({
            frameByteLength: 100 * 1024 * 1024,
            requestedFrameCount: 180,
        })).resolves.toMatchObject({
            frameCount: 3,
            availableByteLength: 512 * 1024 * 1024,
        });
    });

    it("scales the OPFS safety margin down for small browser quotas", async () => {
        const writable = {
            write: vi.fn(async () => {}),
            close: vi.fn(async () => {}),
            abort: vi.fn(async () => {}),
        };
        createMockOpfs(
            writable,
            {
                estimate: async () => ({
                    quota: 64 * 1024 * 1024,
                    usage: 0,
                }),
            },
        );
        const cache = await createSimulationFrameFileCache({ cacheKey: "small-quota" });

        await expect(cache.estimateFrameCapacity({
            frameByteLength: 4.6 * 1024 * 1024,
            requestedFrameCount: 180,
        })).resolves.toMatchObject({
            frameCount: 10,
            availableByteLength: 64 * 1024 * 1024,
        });
    });
});
