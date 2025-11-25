export class GpuStaticMeshBufferManager {
    readonly verticesBuffer: GPUBuffer;
    readonly indicesBuffer: GPUBuffer;
    readonly numIndices: number;

    constructor({
        device,
        vertices,
        indices,
    }: {
        device: GPUDevice,
        vertices: number[],
        indices: number[],
    }) {
        this.numIndices = indices.length;

        const flatVertices = new Float32Array(vertices.length * 4);
        for (let i = 0; i < vertices.length; i++) {
            flatVertices[i] = vertices[i];
        }

        const flatIndices = new Uint32Array(indices.length * 4);
        for (let i = 0; i < indices.length; i++) {
            flatIndices[i] = indices[i];
        }
        
        this.verticesBuffer = device.createBuffer({
            label: "static mesh vertices buffer",
            size: flatVertices.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
        });

        this.indicesBuffer = device.createBuffer({
            label: "static mesh indices buffer",
            size: flatIndices.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDEX,
        });

        device.queue.writeBuffer(
            this.verticesBuffer, 
            0, 
            flatVertices.buffer,
            flatVertices.byteOffset,
            flatVertices.byteLength
        );

        device.queue.writeBuffer(
            this.indicesBuffer, 
            0, 
            flatIndices.buffer,
            flatIndices.byteOffset,
            flatIndices.byteLength
        );
    }
}
