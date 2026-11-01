import { buildUniformSpawnPoints, type SpawnMeshObject } from "./buildUniformSpawnPoints";

export type SpawnMeshSource = {
    vertices: number[][],
    objects?: SpawnMeshObject[],
};

export class GpuSpawnVolumeBufferManager {
    readonly spawnPointsBuffer: GPUBuffer;
    readonly nSpawnPoints: number;

    constructor({
        device,
        nParticles,
        source,
    }: {
        device: GPUDevice,
        nParticles: number,
        source: SpawnMeshSource,
    }) {
        const { points: spawnPoints } = buildUniformSpawnPoints(source.vertices, {
            nPoints: nParticles,
            objects: source.objects,
        });

        if (spawnPoints.length !== nParticles * 4) {
            throw new Error("spawn point buffer must contain one vec4 per particle");
        }

        const spawnPointsBuffer = device.createBuffer({
            label: "uniform spawn points buffer",
            size: Math.max(spawnPoints.byteLength, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        if (spawnPoints.byteLength > 0) {
            device.queue.writeBuffer(
                spawnPointsBuffer,
                0,
                spawnPoints.buffer as ArrayBuffer,
                spawnPoints.byteOffset,
                spawnPoints.byteLength,
            );
        }

        this.spawnPointsBuffer = spawnPointsBuffer;
        this.nSpawnPoints = nParticles;
    }

    destroy() {
        this.spawnPointsBuffer.destroy();
    }
}
