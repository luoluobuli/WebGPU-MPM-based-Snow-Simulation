export const SIMULATION_FRAME_CACHE_DIRECTORY_NAME = "websnow-simulation-frame-cache";

const SIMULATION_FRAME_FILE_NAME_PATTERN = /^frame-(\d{6})\.bin$/;

export const simulationFrameFileName = (frameIndex: number) =>
    `frame-${frameIndex.toString().padStart(6, "0")}.bin`;

export const simulationFrameIndexFromFileName = (fileName: string) => {
    const match = SIMULATION_FRAME_FILE_NAME_PATTERN.exec(fileName);
    if (match === null) return null;

    return Number.parseInt(match[1], 10);
};

export const sanitizeSimulationFrameCacheKey = (cacheKey: string) =>
    cacheKey.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 96);
