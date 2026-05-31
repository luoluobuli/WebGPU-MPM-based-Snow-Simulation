import { buildUniformSpawnPoints, type SpawnMeshObject } from "./buildUniformSpawnPoints";

export class GpuSpawnVolumeBufferManager {
    readonly spawnPointsBuffer: GPUBuffer;
    readonly nSpawnPoints: number;

    constructor({
        device,
        vertices,
        nParticles,
        objects,
    }: {
        device: GPUDevice,
        vertices: number[][],
        nParticles: number,
        objects?: SpawnMeshObject[],
    }) {
        const spawnVolume = buildUniformSpawnPoints(vertices, {
            nPoints: nParticles,
            objects,
        });

        const spawnPointsBuffer = device.createBuffer({
            label: "uniform spawn points buffer",
            size: spawnVolume.points.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(
            spawnPointsBuffer,
            0,
            spawnVolume.points.buffer as ArrayBuffer,
            spawnVolume.points.byteOffset,
            spawnVolume.points.byteLength,
        );

        this.spawnPointsBuffer = spawnPointsBuffer;
        this.nSpawnPoints = spawnVolume.pointCount;
    }
}
