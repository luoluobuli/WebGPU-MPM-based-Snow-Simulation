import type { Mat4 } from "wgpu-matrix";

export interface ColliderGeometry {
    positions: number[];
    normals: number[];
    indices: number[];
}

export class GpuColliderBufferManager {
    readonly colliderDataBuffer: GPUBuffer;
    readonly numIndices: number;
    readonly minCoords: [number, number, number];
    readonly maxCoords: [number, number, number];

    readonly indicesOffset: number;
    readonly verticesOffset: number;
    readonly normalsOffset: number;

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

        // Pack data: [Indices (u32), Vertices (f32), Normals (f32)]
        // All are 4 bytes.
        const totalSize = (indices.length + vertices.length + normals.length) * 4;
        
        // Use a DataView or just write separately with offsets.
        // Actually, creating one large buffer and writing parts is fine.
        
        this.colliderDataBuffer = device.createBuffer({
            label: "collider data buffer",
            size: totalSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX | GPUBufferUsage.INDEX,
        });

        const flatIndices = new Uint32Array(indices);
        const flatIndicesBytes = flatIndices.byteLength;
        
        const flatVertices = new Float32Array(vertices);
        const flatVerticesBytes = flatVertices.byteLength;

        const flatNormals = new Float32Array(normals);
        const flatNormalsBytes = flatNormals.byteLength;

        this.indicesOffset = 0;
        this.verticesOffset = flatIndicesBytes;
        this.normalsOffset = flatIndicesBytes + flatVerticesBytes;

        device.queue.writeBuffer(
            this.colliderDataBuffer,
            0,
            flatIndices.buffer,
            flatIndices.byteOffset,
            flatIndicesBytes
        );

        device.queue.writeBuffer(
            this.colliderDataBuffer,
            flatIndicesBytes,
            flatVertices.buffer,
            flatVertices.byteOffset,
            flatVerticesBytes
        );

        device.queue.writeBuffer(
            this.colliderDataBuffer,
            flatIndicesBytes + flatVerticesBytes,
            flatNormals.buffer,
            flatNormals.byteOffset,
            flatNormalsBytes
        );
        
        // Keep these for potential reuse if needed, or remove them. 
        // The properties were removed from class def, so I won't assign them.
    }
}
