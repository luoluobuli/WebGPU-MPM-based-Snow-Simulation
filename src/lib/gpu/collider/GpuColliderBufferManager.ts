import type { Mat4 } from "wgpu-matrix";

export interface ColliderGeometry {
    positions: number[];
    normals: number[];
    indices: number[];
    objects: {
        min: [number, number, number];
        max: [number, number, number];
        startIndex: number;
        countIndices: number;
    }[];
}

export class GpuColliderBufferManager {
    readonly colliderDataBuffer: GPUBuffer;
    readonly colliderObjectsBuffer: GPUBuffer;
    readonly numIndices: number;
    readonly numObjects: number;
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
        objects,
    }: {
        device: GPUDevice,
        vertices: number[],
        normals: number[],
        indices: number[],
        objects: {
            min: [number, number, number];
            max: [number, number, number];
            startIndex: number;
            countIndices: number;
        }[],
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
        this.numObjects = objects.length;

        // Pack data: [Indices (u32), Vertices (f32), Normals (f32)]
        // All are 4 bytes.
        const totalSize = (indices.length + vertices.length + normals.length) * 4;
        
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

        // Pack Objects
        // Struct: { min: vec3f (12), startIndex: u32 (4), max: vec3f (12), countIndices: u32 (4) } = 32 bytes
        const objectStride = 32;
        const totalObjectsSize = objects.length * objectStride;
        
        this.colliderObjectsBuffer = device.createBuffer({
            label: "collider objects buffer",
            size: totalObjectsSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const objectData = new ArrayBuffer(totalObjectsSize);
        const dataView = new DataView(objectData);

        for (let i = 0; i < objects.length; i++) {
            const obj = objects[i];
            const offset = i * objectStride;
            
            // min (0)
            dataView.setFloat32(offset + 0, obj.min[0], true);
            dataView.setFloat32(offset + 4, obj.min[1], true);
            dataView.setFloat32(offset + 8, obj.min[2], true);
            
            // startIndex (12)
            dataView.setUint32(offset + 12, obj.startIndex, true);
            
            // max (16)
            dataView.setFloat32(offset + 16, obj.max[0], true);
            dataView.setFloat32(offset + 20, obj.max[1], true);
            dataView.setFloat32(offset + 24, obj.max[2], true);
            
            // countIndices (28)
            dataView.setUint32(offset + 28, obj.countIndices, true);
        }

        device.queue.writeBuffer(
            this.colliderObjectsBuffer,
            0,
            objectData
        );
    }
}
