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

// BVH Node structure (32 bytes, 8 floats/uints):
// min: vec3f (12 bytes)
// leftChildOrPrimIndex: u32 (4 bytes) - If leaf: start index of triangle. If internal: left child index.
// max: vec3f (12 bytes)
// rightChildOrPrimCount: u32 (4 bytes) - If leaf: primitive count. If internal: right child index.
interface BvhNode {
    minX: number;
    minY: number;
    minZ: number;
    leftChildOrPrimIndex: number;
    maxX: number;
    maxY: number;
    maxZ: number;
    rightChildOrPrimCount: number;
    isLeaf: boolean; // Not stored in GPU, used during construction
}

interface Triangle {
    idx0: number;
    idx1: number;
    idx2: number;
    centroid: [number, number, number];
    minBounds: [number, number, number];
    maxBounds: [number, number, number];
}

export class GpuColliderBufferManager {
    readonly colliderDataBuffer: GPUBuffer;
    readonly colliderBvhBuffer: GPUBuffer;
    readonly numIndices: number;
    readonly numBvhNodes: number;
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

        // Build BVH from triangles
        const { nodes, triangleIndices } = this.buildBvh(vertices, indices);
        this.numBvhNodes = nodes.length;

        // Create BVH buffer (32 bytes per node)
        const bvhData = new ArrayBuffer(nodes.length * 32);
        const bvhFloatView = new Float32Array(bvhData);
        const bvhUintView = new Uint32Array(bvhData);

        for (let i = 0; i < nodes.length; i++) {
            const node = nodes[i];
            const baseIdx = i * 8;
            bvhFloatView[baseIdx + 0] = node.minX;
            bvhFloatView[baseIdx + 1] = node.minY;
            bvhFloatView[baseIdx + 2] = node.minZ;
            bvhUintView[baseIdx + 3] = node.leftChildOrPrimIndex;
            bvhFloatView[baseIdx + 4] = node.maxX;
            bvhFloatView[baseIdx + 5] = node.maxY;
            bvhFloatView[baseIdx + 6] = node.maxZ;
            // For leaves: store primitive count with high bit set to mark as leaf
            // For internal: store right child index (high bit clear)
            bvhUintView[baseIdx + 7] = node.isLeaf 
                ? (node.rightChildOrPrimCount | 0x80000000) 
                : node.rightChildOrPrimCount;
        }

        this.colliderBvhBuffer = device.createBuffer({
            label: "collider BVH buffer",
            size: Math.max(bvhData.byteLength, 32), // Minimum 32 bytes
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.colliderBvhBuffer, 0, bvhData);

        // Rewrite indices in BVH order (triangles reordered by BVH leaf order)
        const reorderedIndices = new Uint32Array(triangleIndices.length * 3);
        for (let i = 0; i < triangleIndices.length; i++) {
            const triIdx = triangleIndices[i];
            reorderedIndices[i * 3 + 0] = indices[triIdx * 3 + 0];
            reorderedIndices[i * 3 + 1] = indices[triIdx * 3 + 1];
            reorderedIndices[i * 3 + 2] = indices[triIdx * 3 + 2];
        }
        device.queue.writeBuffer(this.colliderDataBuffer, 0, reorderedIndices);
    }

    private buildBvh(vertices: number[], indices: number[]): { nodes: BvhNode[], triangleIndices: number[] } {
        const numTriangles = indices.length / 3;
        
        if (numTriangles === 0) {
            // Return a dummy leaf node for empty meshes
            return {
                nodes: [{
                    minX: 0, minY: 0, minZ: 0, leftChildOrPrimIndex: 0,
                    maxX: 0, maxY: 0, maxZ: 0, rightChildOrPrimCount: 0,
                    isLeaf: true
                }],
                triangleIndices: []
            };
        }

        // Build triangle data with centroids and bounds
        const triangles: Triangle[] = [];
        for (let i = 0; i < numTriangles; i++) {
            const idx0 = indices[i * 3 + 0];
            const idx1 = indices[i * 3 + 1];
            const idx2 = indices[i * 3 + 2];

            const v0 = [vertices[idx0 * 3], vertices[idx0 * 3 + 1], vertices[idx0 * 3 + 2]];
            const v1 = [vertices[idx1 * 3], vertices[idx1 * 3 + 1], vertices[idx1 * 3 + 2]];
            const v2 = [vertices[idx2 * 3], vertices[idx2 * 3 + 1], vertices[idx2 * 3 + 2]];

            const minBounds: [number, number, number] = [
                Math.min(v0[0], v1[0], v2[0]),
                Math.min(v0[1], v1[1], v2[1]),
                Math.min(v0[2], v1[2], v2[2])
            ];
            const maxBounds: [number, number, number] = [
                Math.max(v0[0], v1[0], v2[0]),
                Math.max(v0[1], v1[1], v2[1]),
                Math.max(v0[2], v1[2], v2[2])
            ];
            const centroid: [number, number, number] = [
                (v0[0] + v1[0] + v2[0]) / 3,
                (v0[1] + v1[1] + v2[1]) / 3,
                (v0[2] + v1[2] + v2[2]) / 3
            ];

            triangles.push({ idx0, idx1, idx2, centroid, minBounds, maxBounds });
        }

        // Build BVH recursively
        const nodes: BvhNode[] = [];
        const triangleIndices: number[] = Array.from({ length: numTriangles }, (_, i) => i);

        this.buildBvhRecursive(triangles, triangleIndices, 0, numTriangles, nodes);

        return { nodes, triangleIndices };
    }

    private buildBvhRecursive(
        triangles: Triangle[],
        triangleIndices: number[],
        start: number,
        end: number,
        nodes: BvhNode[]
    ): number {
        const nodeIndex = nodes.length;
        const count = end - start;

        // Compute bounds for this node
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        for (let i = start; i < end; i++) {
            const tri = triangles[triangleIndices[i]];
            minX = Math.min(minX, tri.minBounds[0]);
            minY = Math.min(minY, tri.minBounds[1]);
            minZ = Math.min(minZ, tri.minBounds[2]);
            maxX = Math.max(maxX, tri.maxBounds[0]);
            maxY = Math.max(maxY, tri.maxBounds[1]);
            maxZ = Math.max(maxZ, tri.maxBounds[2]);
        }

        // Leaf node threshold
        const MAX_LEAF_SIZE = 4;
        if (count <= MAX_LEAF_SIZE) {
            nodes.push({
                minX, minY, minZ,
                leftChildOrPrimIndex: start,
                maxX, maxY, maxZ,
                rightChildOrPrimCount: count,
                isLeaf: true
            });
            return nodeIndex;
        }

        // Find best split axis using SAH (Surface Area Heuristic) approximation
        const extentX = maxX - minX;
        const extentY = maxY - minY;
        const extentZ = maxZ - minZ;

        let splitAxis = 0;
        if (extentY > extentX && extentY > extentZ) splitAxis = 1;
        else if (extentZ > extentX && extentZ > extentY) splitAxis = 2;

        // Sort triangles by centroid along split axis
        const subIndices = triangleIndices.slice(start, end);
        subIndices.sort((a, b) => triangles[a].centroid[splitAxis] - triangles[b].centroid[splitAxis]);
        for (let i = 0; i < subIndices.length; i++) {
            triangleIndices[start + i] = subIndices[i];
        }

        // Split at median
        const mid = start + Math.floor(count / 2);

        // Create internal node (placeholder, will update children after recursion)
        nodes.push({
            minX, minY, minZ,
            leftChildOrPrimIndex: 0, // Will be updated
            maxX, maxY, maxZ,
            rightChildOrPrimCount: 0, // Will be updated
            isLeaf: false
        });

        // Build children - left child is always the next node after this one
        const leftChild = this.buildBvhRecursive(triangles, triangleIndices, start, mid, nodes);
        const rightChild = this.buildBvhRecursive(triangles, triangleIndices, mid, end, nodes);

        // Update internal node with child indices
        nodes[nodeIndex].leftChildOrPrimIndex = leftChild;
        nodes[nodeIndex].rightChildOrPrimCount = rightChild;

        return nodeIndex;
    }
}
