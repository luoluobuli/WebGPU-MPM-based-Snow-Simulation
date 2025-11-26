import type { Mat4 } from "wgpu-matrix";

export interface ColliderGeometry {
    positions: number[];
    normals: number[];
    indices: number[];
    transform: Mat4;
}

export class GpuColliderBufferManager {
    readonly verticesBuffer: GPUBuffer;
    readonly normalsBuffer: GPUBuffer;
    readonly indicesBuffer: GPUBuffer;
    readonly numIndices: number;
    readonly minCoords: [number, number, number];
    readonly maxCoords: [number, number, number];

    constructor({
        device,
        vertices,
        normals,
        indices,
    }: {
        device: GPUDevice,
        vertices: number[],
        normals: number[],
        indices: number[],
    }) {
        // tmp stores bounding box as the collider; will use the geometry itself in the future
        const min: [number, number, number] = [Infinity, Infinity, Infinity];
        const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
        for (let i = 0; i < vertices.length; i+=3) {
            min[0] = Math.min(min[0], vertices[i]);
            min[1] = Math.min(min[1], vertices[i+1]);
            min[2] = Math.min(min[2], vertices[i+2]);

            max[0] = Math.max(max[0], vertices[i]);
            max[1] = Math.max(max[1], vertices[i+1]);
            max[2] = Math.max(max[2], vertices[i+2]);
        }
        this.minCoords = min;
        this.maxCoords = max;

        this.numIndices = indices.length;

        const flatVertices = new Float32Array(vertices.length * 4);
        for (let i = 0; i < vertices.length; i++) {
            flatVertices[i] = vertices[i];
        }

        const flatNormals = new Float32Array(normals.length * 4);
        for (let i = 0; i < normals.length; i++) {
            flatNormals[i] = normals[i];
        }

        const flatIndices = new Uint32Array(indices.length * 4);
        for (let i = 0; i < indices.length; i++) {
            flatIndices[i] = indices[i];
        }
        
        this.verticesBuffer = device.createBuffer({
            label: "collider vertices buffer",
            size: flatVertices.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
        });

        this.normalsBuffer = device.createBuffer({
            label: "collider normals buffer",
            size: flatNormals.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
        });

        this.indicesBuffer = device.createBuffer({
            label: "collider indices buffer",
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
            this.normalsBuffer, 
            0, 
            flatNormals.buffer,
            flatNormals.byteOffset,
            flatNormals.byteLength
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
