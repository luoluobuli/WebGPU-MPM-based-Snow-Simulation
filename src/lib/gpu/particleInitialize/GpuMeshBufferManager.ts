export class GpuMeshBufferManager {
    readonly meshVerticesBuffer: GPUBuffer;
    readonly numVertices: number;
    readonly minCoords: [number, number, number];
    readonly maxCoords: [number, number, number];

    constructor({
        device,
        vertices,
    }: {
        device: GPUDevice,
        vertices: number[][],
    }) {
        const min: [number, number, number] = [Infinity, Infinity, Infinity];
        const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
        
        for (const vert of vertices) {
            min[0] = Math.min(min[0], vert[0]);
            min[1] = Math.min(min[1], vert[1]);
            min[2] = Math.min(min[2], vert[2]);

            max[0] = Math.max(max[0], vert[0]);
            max[1] = Math.max(max[1], vert[1]);
            max[2] = Math.max(max[2], vert[2]);
        }
        
        this.minCoords = min;
        this.maxCoords = max;
        
        const flatVertices = new Float32Array(vertices.length * 4);
        for (let i = 0; i < vertices.length; i++) {
            flatVertices[i * 4] = vertices[i][0];
            flatVertices[i * 4 + 1] = vertices[i][1];
            flatVertices[i * 4 + 2] = vertices[i][2];
        }

        this.meshVerticesBuffer = device.createBuffer({
            label: "mesh vertices buffer",
            size: flatVertices.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(this.meshVerticesBuffer, 0, flatVertices);
        this.numVertices = vertices.length;
    }
}
