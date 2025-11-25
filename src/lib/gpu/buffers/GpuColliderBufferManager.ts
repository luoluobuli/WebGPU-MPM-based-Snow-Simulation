export class GpuColliderBufferManager {
    readonly colliderVerticesBuffer: GPUBuffer;
    readonly colliderIndicesBuffer: GPUBuffer;
    readonly numIndices: number;

    constructor({
        device,
        vertices,
        indices,
    }: {
        device: GPUDevice,
        vertices: Float32Array,
        indices: Uint32Array,
    }) {
        this.numIndices = indices.length;
        
        this.colliderVerticesBuffer = device.createBuffer({
            label: "collider vertices buffer",
            size: vertices.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
        });

        this.colliderIndicesBuffer = device.createBuffer({
            label: "collider indices buffer",
            size: indices.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDEX,
        });

        device.queue.writeBuffer(
            this.colliderVerticesBuffer, 
            0, 
            vertices.buffer,
            vertices.byteOffset,
            vertices.byteLength
        );

        device.queue.writeBuffer(
            this.colliderIndicesBuffer, 
            0, 
            indices.buffer,
            indices.byteOffset,
            indices.byteLength
        );
    }
}
