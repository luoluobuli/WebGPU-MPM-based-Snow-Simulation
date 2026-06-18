export const SIMULATION_FRAME_CACHE_DIRECTORY_NAME = "websnow-simulation-frame-cache";

export const simulationFrameFileName = (frameIndex: number) =>
    `frame-${frameIndex.toString().padStart(6, "0")}.bin`;

export const sanitizeSimulationFrameCacheKey = (cacheKey: string) =>
    cacheKey.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 96);
