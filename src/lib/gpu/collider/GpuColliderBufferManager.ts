import type { Mat4 } from "wgpu-matrix";

export interface ColliderGeometry {
    positions: number[];
    normals: number[];
    uvs: number[];
    materialIndices: number[];
    textures: ImageBitmap[];
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
    readonly numIndices: number;
    readonly minCoords: [number, number, number];
    readonly maxCoords: [number, number, number];

    readonly indicesOffset: number;
    readonly verticesOffset: number;
    readonly normalsOffset: number;
    readonly uvsOffset: number;
    readonly materialIndicesOffset: number;
    
    readonly textureArray: GPUTexture | null;
    readonly textureArrayView: GPUTextureView | null;
    readonly sampler: GPUSampler;
    readonly numTextures: number;

    constructor({
        device,
        vertices,
        normals,
        uvs,
        materialIndices,
        textures,
        indices,
    }: {
        device: GPUDevice,
        vertices: number[],
        normals: number[],
        uvs: number[],
        materialIndices: number[],
        textures: ImageBitmap[],
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
        this.numTextures = textures.length;

        // Pack data: [Indices (u32), Vertices (f32x3), Normals (f32x3), UVs (f32x2), MaterialIndices (u32)]
        const indicesSize = indices.length * 4;
        const verticesSize = vertices.length * 4;
        const normalsSize = normals.length * 4;
        const uvsSize = uvs.length * 4;
        const materialIndicesSize = materialIndices.length * 4;
        const totalSize = indicesSize + verticesSize + normalsSize + uvsSize + materialIndicesSize;
        
        this.colliderDataBuffer = device.createBuffer({
            label: "collider data buffer",
            size: totalSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX | GPUBufferUsage.INDEX,
        });

        const flatIndices = new Uint32Array(indices);
        const flatVertices = new Float32Array(vertices);
        const flatNormals = new Float32Array(normals);
        const flatUvs = new Float32Array(uvs);
        const flatMaterialIndices = new Uint32Array(materialIndices);

        this.indicesOffset = 0;
        this.verticesOffset = indicesSize;
        this.normalsOffset = indicesSize + verticesSize;
        this.uvsOffset = indicesSize + verticesSize + normalsSize;
        this.materialIndicesOffset = indicesSize + verticesSize + normalsSize + uvsSize;

        device.queue.writeBuffer(this.colliderDataBuffer, 0, flatIndices);
        device.queue.writeBuffer(this.colliderDataBuffer, indicesSize, flatVertices);
        device.queue.writeBuffer(this.colliderDataBuffer, indicesSize + verticesSize, flatNormals);
        device.queue.writeBuffer(this.colliderDataBuffer, indicesSize + verticesSize + normalsSize, flatUvs);
        device.queue.writeBuffer(this.colliderDataBuffer, indicesSize + verticesSize + normalsSize + uvsSize, flatMaterialIndices);

        // Create sampler
        this.sampler = device.createSampler({
            magFilter: "linear",
            minFilter: "linear",
            mipmapFilter: "linear",
            addressModeU: "repeat",
            addressModeV: "repeat",
        });

        // Create texture array from ImageBitmaps
        if (textures.length > 0) {
            const textureSize = 256; // Resize all textures to a common size
            
            this.textureArray = device.createTexture({
                label: "collider texture array",
                size: [textureSize, textureSize, textures.length],
                format: "rgba8unorm",
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
            });
            
            for (let i = 0; i < textures.length; i++) {
                device.queue.copyExternalImageToTexture(
                    { source: textures[i] },
                    { texture: this.textureArray, origin: [0, 0, i] },
                    [textures[i].width, textures[i].height]
                );
            }
            
            this.textureArrayView = this.textureArray.createView({
                dimension: "2d-array",
            });
        } else {
            // Create a dummy 1x1 white texture if no textures
            this.textureArray = device.createTexture({
                label: "collider dummy texture",
                size: [1, 1, 1],
                format: "rgba8unorm",
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            device.queue.writeTexture(
                { texture: this.textureArray },
                new Uint8Array([255, 255, 255, 255]),
                { bytesPerRow: 4 },
                [1, 1, 1]
            );
            this.textureArrayView = this.textureArray.createView({
                dimension: "2d-array",
            });
        }
    }
}
