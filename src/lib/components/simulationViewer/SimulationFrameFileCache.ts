import {
    SIMULATION_FRAME_CACHE_DIRECTORY_NAME,
    simulationFrameIndexFromFileName,
    simulationFrameFileName,
    sanitizeSimulationFrameCacheKey,
} from "./SimulationFrameCacheNames";

export type SimulationFrameFileCache = {
    storageLabel: string,
    clear: () => Promise<void>,
    estimateFrameCapacity: (options: {
        frameByteLength: number,
        requestedFrameCount: number,
    }) => Promise<SimulationFrameCacheCapacity>,
    findNextUncachedFrame?: (options: {
        frameByteLength: number,
        requestedFrameCount: number,
    }) => Promise<number>,
    readFrame: (frameIndex: number) => Promise<ArrayBuffer | null>,
    writeFrame: (frameIndex: number, snapshot: ArrayBuffer) => Promise<void>,
};

export type SimulationFrameCacheCapacity = {
    frameCount: number,
    availableByteLength: number | null,
    quotaByteLength: number | null,
    usageByteLength: number | null,
};

type DirectoryPickerGlobal = typeof globalThis & {
    showDirectoryPicker?: (options?: {
        id?: string,
        mode?: "read" | "readwrite",
    }) => Promise<FileSystemDirectoryHandle>,
};

type WritableDirectoryHandle = FileSystemDirectoryHandle & {
    queryPermission?: (descriptor?: {
        mode?: "read" | "readwrite",
    }) => Promise<PermissionState>,
    requestPermission?: (descriptor?: {
        mode?: "read" | "readwrite",
    }) => Promise<PermissionState>,
};

const OPFS_WRITE_CHUNK_BYTE_LENGTH = 8 * 1024 * 1024;
const OPFS_CACHE_MAX_SAFETY_MARGIN_BYTE_LENGTH = 128 * 1024 * 1024;
const OPFS_CACHE_SAFETY_MARGIN_QUOTA_FRACTION = 0.25;

const calculateOpfsCacheSafetyMarginByteLength = (quotaByteLength: number) =>
    Math.floor(Math.min(
        OPFS_CACHE_MAX_SAFETY_MARGIN_BYTE_LENGTH,
        quotaByteLength * OPFS_CACHE_SAFETY_MARGIN_QUOTA_FRACTION,
    ));

const safeRequestedFrameCount = (requestedFrameCount: number) =>
    Number.isFinite(requestedFrameCount)
        ? Math.max(0, Math.floor(requestedFrameCount))
        : 0;

const getDirectoryPickerGlobal = () => globalThis as DirectoryPickerGlobal;

const ensureDirectoryReadWritePermission = async (
    directory: FileSystemDirectoryHandle,
) => {
    const writableDirectory = directory as WritableDirectoryHandle;
    if (
        writableDirectory.queryPermission === undefined
        || writableDirectory.requestPermission === undefined
    ) {
        return;
    }

    const descriptor = { mode: "readwrite" } as const;
    const currentPermission = await writableDirectory.queryPermission(descriptor);
    if (currentPermission === "granted") return;

    const requestedPermission = await writableDirectory.requestPermission(descriptor);
    if (requestedPermission !== "granted") {
        throw new DOMException(
            "Cache directory write permission was not granted",
            "NotAllowedError",
        );
    }
};

const getExistingDirectoryHandle = async (
    directory: FileSystemDirectoryHandle,
    name: string,
) => {
    try {
        return await directory.getDirectoryHandle(name);
    } catch (error) {
        if (error instanceof DOMException && error.name === "NotFoundError") {
            return null;
        }

        throw error;
    }
};

const directoryContainsSimulationFrameFiles = async (
    directory: FileSystemDirectoryHandle,
) => {
    for await (const [name] of directory.entries()) {
        if (simulationFrameIndexFromFileName(name) !== null) {
            return true;
        }
    }

    return false;
};

const openFrameDirectory = async ({
    rootDirectory,
    cacheKey,
}: {
    rootDirectory: FileSystemDirectoryHandle,
    cacheKey: string,
}) => {
    if (await directoryContainsSimulationFrameFiles(rootDirectory)) {
        return rootDirectory;
    }

    const sanitizedCacheKey = sanitizeSimulationFrameCacheKey(cacheKey);
    const selectedCacheKeyDirectory = await getExistingDirectoryHandle(
        rootDirectory,
        sanitizedCacheKey,
    );
    if (selectedCacheKeyDirectory !== null) {
        return selectedCacheKeyDirectory;
    }

    const cacheRoot = await rootDirectory.getDirectoryHandle(
        SIMULATION_FRAME_CACHE_DIRECTORY_NAME,
        { create: true },
    );

    for await (const [name] of cacheRoot.entries()) {
        if (name !== sanitizedCacheKey) {
            await cacheRoot.removeEntry(name, { recursive: true });
        }
    }

    return await cacheRoot.getDirectoryHandle(
        sanitizedCacheKey,
        { create: true },
    );
};

const estimateSelectedDirectoryFrameCapacity = async ({
    requestedFrameCount,
}: {
    frameByteLength: number,
    requestedFrameCount: number,
}): Promise<SimulationFrameCacheCapacity> => ({
    frameCount: safeRequestedFrameCount(requestedFrameCount),
    availableByteLength: null,
    quotaByteLength: null,
    usageByteLength: null,
});

const findDirectoryNextUncachedFrame = async ({
    directory,
    frameByteLength,
    requestedFrameCount,
}: {
    directory: FileSystemDirectoryHandle,
    frameByteLength: number,
    requestedFrameCount: number,
}) => {
    const frameCount = safeRequestedFrameCount(requestedFrameCount);
    if (!Number.isFinite(frameByteLength) || frameByteLength <= 0) {
        return 0;
    }

    const existingFrameIndexes = new Set<number>();
    for await (const [name] of directory.entries()) {
        const frameIndex = simulationFrameIndexFromFileName(name);
        if (
            frameIndex !== null
            && frameIndex >= 0
            && frameIndex < frameCount
        ) {
            existingFrameIndexes.add(frameIndex);
        }
    }

    for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
        if (!existingFrameIndexes.has(frameIndex)) {
            return frameIndex;
        }

        const handle = await directory.getFileHandle(
            simulationFrameFileName(frameIndex),
        );
        const file = await handle.getFile();
        if (file.size !== frameByteLength) {
            return frameIndex;
        }
    }

    return frameCount;
};

export const isQuotaExceededError = (error: unknown) => {
    if (!(error instanceof DOMException)) return false;

    return error.name === "QuotaExceededError"
        || error.message.toLowerCase().includes("quota");
};

export const canSelectSimulationFrameCacheDirectory = () =>
    getDirectoryPickerGlobal().showDirectoryPicker !== undefined;

export const selectSimulationFrameCacheDirectory = async () => {
    const directoryPickerGlobal = getDirectoryPickerGlobal();
    if (directoryPickerGlobal.showDirectoryPicker === undefined) {
        throw new Error("This browser cannot select a cache folder");
    }

    const directory = await directoryPickerGlobal.showDirectoryPicker({
        id: "websnow-simulation-frame-cache",
        mode: "readwrite",
    });
    await ensureDirectoryReadWritePermission(directory);

    return directory;
};

class DirectorySimulationFrameFileCache implements SimulationFrameFileCache {
    readonly storageLabel: string;

    private readonly directory: FileSystemDirectoryHandle;
    private readonly estimateFrameCapacityImpl: (
        options: {
            frameByteLength: number,
            requestedFrameCount: number,
        },
    ) => Promise<SimulationFrameCacheCapacity>;

    constructor({
        directory,
        storageLabel,
        estimateFrameCapacity,
    }: {
        directory: FileSystemDirectoryHandle,
        storageLabel: string,
        estimateFrameCapacity: (
            options: {
                frameByteLength: number,
                requestedFrameCount: number,
            },
        ) => Promise<SimulationFrameCacheCapacity>,
    }) {
        this.directory = directory;
        this.storageLabel = storageLabel;
        this.estimateFrameCapacityImpl = estimateFrameCapacity;
    }

    async clear() {
        for await (const [name] of this.directory.entries()) {
            await this.directory.removeEntry(name, { recursive: true });
        }
    }

    async estimateFrameCapacity({
        frameByteLength,
        requestedFrameCount,
    }: {
        frameByteLength: number,
        requestedFrameCount: number,
    }) {
        return await this.estimateFrameCapacityImpl({
            frameByteLength,
            requestedFrameCount,
        });
    }

    async findNextUncachedFrame({
        frameByteLength,
        requestedFrameCount,
    }: {
        frameByteLength: number,
        requestedFrameCount: number,
    }) {
        return await findDirectoryNextUncachedFrame({
            directory: this.directory,
            frameByteLength,
            requestedFrameCount,
        });
    }

    async readFrame(frameIndex: number) {
        try {
            const handle = await this.directory.getFileHandle(
                simulationFrameFileName(frameIndex),
            );
            const file = await handle.getFile();

            return await file.arrayBuffer();
        } catch (error) {
            if (error instanceof DOMException && error.name === "NotFoundError") {
                return null;
            }

            throw error;
        }
    }

    async writeFrame(frameIndex: number, snapshot: ArrayBuffer) {
        const name = simulationFrameFileName(frameIndex);
        const handle = await this.directory.getFileHandle(
            name,
            { create: true },
        );
        const writable = await handle.createWritable();

        try {
            for (
                let offset = 0;
                offset < snapshot.byteLength;
                offset += OPFS_WRITE_CHUNK_BYTE_LENGTH
            ) {
                const byteLength = Math.min(
                    OPFS_WRITE_CHUNK_BYTE_LENGTH,
                    snapshot.byteLength - offset,
                );

                await writable.write(new Uint8Array(snapshot, offset, byteLength));
            }

            await writable.close();
        } catch (error) {
            await writable.abort().catch(() => {});
            await this.directory.removeEntry(name).catch(() => {});

            throw error;
        }
    }
}

const estimateOpfsFrameCapacity = async ({
    frameByteLength,
    requestedFrameCount,
}: {
    frameByteLength: number,
    requestedFrameCount: number,
}): Promise<SimulationFrameCacheCapacity> => {
    const requestedFrameCapacity = safeRequestedFrameCount(requestedFrameCount);

    if (!Number.isFinite(frameByteLength) || frameByteLength <= 0) {
        return {
            frameCount: requestedFrameCapacity,
            availableByteLength: null,
            quotaByteLength: null,
            usageByteLength: null,
        };
    }

    const estimate = await navigator.storage?.estimate?.().catch(() => null);
    const quotaByteLength = estimate?.quota ?? null;
    const usageByteLength = estimate?.usage ?? null;

    if (quotaByteLength === null || usageByteLength === null) {
        return {
            frameCount: requestedFrameCapacity,
            availableByteLength: null,
            quotaByteLength,
            usageByteLength,
        };
    }

    const availableByteLength = Math.max(0, quotaByteLength - usageByteLength);
    const safetyMarginByteLength = calculateOpfsCacheSafetyMarginByteLength(
        quotaByteLength,
    );
    const writableByteLength = Math.max(
        0,
        availableByteLength - safetyMarginByteLength,
    );

    return {
        frameCount: Math.min(
            requestedFrameCapacity,
            Math.floor(writableByteLength / frameByteLength),
        ),
        availableByteLength,
        quotaByteLength,
        usageByteLength,
    };
};

class MemorySimulationFrameFileCache implements SimulationFrameFileCache {
    readonly storageLabel = "memory cache fallback";

    private readonly frames = new Map<number, ArrayBuffer>();

    async clear() {
        this.frames.clear();
    }

    async estimateFrameCapacity({
        requestedFrameCount,
    }: {
        frameByteLength: number,
        requestedFrameCount: number,
    }) {
        return {
            frameCount: Number.isFinite(requestedFrameCount)
                ? Math.max(0, Math.floor(requestedFrameCount))
                : 0,
            availableByteLength: null,
            quotaByteLength: null,
            usageByteLength: null,
        };
    }

    async findNextUncachedFrame({
        frameByteLength,
        requestedFrameCount,
    }: {
        frameByteLength: number,
        requestedFrameCount: number,
    }) {
        const frameCount = safeRequestedFrameCount(requestedFrameCount);
        if (!Number.isFinite(frameByteLength) || frameByteLength <= 0) {
            return 0;
        }

        for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
            const frame = this.frames.get(frameIndex);
            if (frame === undefined || frame.byteLength !== frameByteLength) {
                return frameIndex;
            }
        }

        return frameCount;
    }

    async readFrame(frameIndex: number) {
        return this.frames.get(frameIndex)?.slice(0) ?? null;
    }

    async writeFrame(frameIndex: number, snapshot: ArrayBuffer) {
        this.frames.set(frameIndex, snapshot.slice(0));
    }
}

export const createSimulationFrameFileCache = async ({
    cacheKey,
    rootDirectory = null,
}: {
    cacheKey: string,
    rootDirectory?: FileSystemDirectoryHandle | null,
}): Promise<SimulationFrameFileCache> => {
    if (rootDirectory !== null) {
        await ensureDirectoryReadWritePermission(rootDirectory);
        const directory = await openFrameDirectory({
            rootDirectory,
            cacheKey,
        });

        return new DirectorySimulationFrameFileCache({
            directory,
            storageLabel: `folder cache: ${rootDirectory.name || "selected folder"}`,
            estimateFrameCapacity: estimateSelectedDirectoryFrameCapacity,
        });
    }

    if (typeof navigator !== "undefined" && navigator.storage?.getDirectory !== undefined) {
        const root = await navigator.storage.getDirectory();
        const directory = await openFrameDirectory({
            rootDirectory: root,
            cacheKey,
        });

        return new DirectorySimulationFrameFileCache({
            directory,
            storageLabel: "OPFS file cache",
            estimateFrameCapacity: estimateOpfsFrameCapacity,
        });
    }

    return new MemorySimulationFrameFileCache();
};
