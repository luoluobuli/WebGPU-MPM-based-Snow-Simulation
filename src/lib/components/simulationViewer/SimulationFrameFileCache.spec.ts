import { afterEach, describe, expect, it, vi } from "vitest";

import {
    canSelectSimulationFrameCacheDirectory,
    createSimulationFrameFileCache,
    selectSimulationFrameCacheDirectory,
} from "./SimulationFrameFileCache";

type MockWritable = {
    write: ReturnType<typeof vi.fn>,
    close: ReturnType<typeof vi.fn>,
    abort: ReturnType<typeof vi.fn>,
};

const createMockOpfs = (
    writable: MockWritable,
    {
        cacheRootEntries = [],
        estimate = async () => ({
            quota: 512 * 1024 * 1024,
            usage: 0,
        }),
    }: {
        cacheRootEntries?: string[],
        estimate?: () => Promise<StorageEstimate>,
    } = {},
) => {
    const frameDirectory = {
        entries: async function* () {},
        removeEntry: vi.fn(async () => {}),
        getFileHandle: vi.fn(async () => ({
            createWritable: vi.fn(async () => writable),
            getFile: vi.fn(async () => ({
                arrayBuffer: vi.fn(async () => new ArrayBuffer(0)),
            })),
        })),
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
        getDirectoryHandle: vi.fn(async () => cacheRootDirectory),
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
    }: {
        cacheRootEntries?: string[],
    } = {},
) => {
    const frameDirectory = {
        entries: async function* () {},
        removeEntry: vi.fn(async () => {}),
        getFileHandle: vi.fn(async () => ({
            createWritable: vi.fn(async () => writable),
            getFile: vi.fn(async () => ({
                arrayBuffer: vi.fn(async () => new ArrayBuffer(0)),
            })),
        })),
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
        getDirectoryHandle: vi.fn(async () => cacheRootDirectory),
    } as unknown as FileSystemDirectoryHandle;

    return {
        cacheRootDirectory,
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
