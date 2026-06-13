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

export const COLLIDER_SDF_RESOLUTION = 64;
export const COLLIDER_SDF_SURFACE_THICKNESS = 0.05;

type Vec3 = [number, number, number];

interface Triangle {
    v0: Vec3;
    v1: Vec3;
    v2: Vec3;
    centroid: Vec3;
    minBounds: Vec3;
    maxBounds: Vec3;
}

interface SdfBvhNode {
    minBounds: Vec3;
    maxBounds: Vec3;
    leftChild: number;
    rightChild: number;
    start: number;
    count: number;
    isLeaf: boolean;
}

export class GpuColliderBufferManager {
    readonly colliderDataBuffer: GPUBuffer;
    readonly colliderSdfBuffer: GPUBuffer;
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
        if (vertices.length === 0 || indices.length === 0) {
            this.minCoords = [0, 0, 0];
            this.maxCoords = [0, 0, 0];
            this.numIndices = 0;
            this.numTextures = 0;
            this.indicesOffset = 0;
            this.verticesOffset = 0;
            this.normalsOffset = 0;
            this.uvsOffset = 0;
            this.materialIndicesOffset = 0;

            this.colliderDataBuffer = device.createBuffer({
                label: "disabled collider data buffer",
                size: 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX | GPUBufferUsage.INDEX,
            });

            this.sampler = device.createSampler({
                magFilter: "linear",
                minFilter: "linear",
                mipmapFilter: "linear",
                addressModeU: "repeat",
                addressModeV: "repeat",
            });

            this.textureArray = device.createTexture({
                label: "disabled collider dummy texture",
                size: [1, 1, 1],
                format: "rgba8unorm",
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            device.queue.writeTexture(
                { texture: this.textureArray },
                new Uint8Array([255, 255, 255, 255]),
                { bytesPerRow: 4 },
                [1, 1, 1],
            );
            this.textureArrayView = this.textureArray.createView({
                dimension: "2d-array",
            });

            const disabledSdf = new Float32Array([1e6]);
            this.colliderSdfBuffer = device.createBuffer({
                label: "disabled collider SDF buffer",
                size: disabledSdf.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(this.colliderSdfBuffer, 0, disabledSdf);

            return;
        }

        const meshMin: Vec3 = [Infinity, Infinity, Infinity];
        const meshMax: Vec3 = [-Infinity, -Infinity, -Infinity];
        for (let i = 0; i < vertices.length; i+=3) {
            meshMin[0] = Math.min(meshMin[0], vertices[i]);
            meshMin[1] = Math.min(meshMin[1], vertices[i+1]);
            meshMin[2] = Math.min(meshMin[2], vertices[i+2]);

            meshMax[0] = Math.max(meshMax[0], vertices[i]);
            meshMax[1] = Math.max(meshMax[1], vertices[i+1]);
            meshMax[2] = Math.max(meshMax[2], vertices[i+2]);
        }

        const sdfBounds = this.buildSdfBounds(meshMin, meshMax);
        this.minCoords = sdfBounds.min;
        this.maxCoords = sdfBounds.max;

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

        const sdfData = this.buildColliderSdf(vertices, indices, sdfBounds);
        this.colliderSdfBuffer = device.createBuffer({
            label: "collider SDF buffer",
            size: Math.max(sdfData.byteLength, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.colliderSdfBuffer, 0, sdfData.buffer as ArrayBuffer, sdfData.byteOffset, sdfData.byteLength);
    }

    private buildSdfBounds(meshMin: Vec3, meshMax: Vec3): { min: Vec3, max: Vec3 } {
        const extent: Vec3 = [
            meshMax[0] - meshMin[0],
            meshMax[1] - meshMin[1],
            meshMax[2] - meshMin[2],
        ];
        const maxExtent = Math.max(extent[0], extent[1], extent[2], 1e-3);
        const voxelEstimate = maxExtent / (COLLIDER_SDF_RESOLUTION - 1);
        const padding = Math.max(COLLIDER_SDF_SURFACE_THICKNESS * 2, voxelEstimate * 2);

        return {
            min: [meshMin[0] - padding, meshMin[1] - padding, meshMin[2] - padding],
            max: [meshMax[0] + padding, meshMax[1] + padding, meshMax[2] + padding],
        };
    }

    private buildColliderSdf(
        vertices: number[],
        indices: number[],
        bounds: { min: Vec3, max: Vec3 },
    ): Float32Array {
        const voxelCount = COLLIDER_SDF_RESOLUTION * COLLIDER_SDF_RESOLUTION * COLLIDER_SDF_RESOLUTION;
        const sdfData = new Float32Array(voxelCount);
        const { triangles, triangleIndices, nodes } = this.buildSdfBvh(vertices, indices);

        if (triangles.length === 0) {
            sdfData.fill(1e6);
            return sdfData;
        }

        const dx = (bounds.max[0] - bounds.min[0]) / (COLLIDER_SDF_RESOLUTION - 1);
        const dy = (bounds.max[1] - bounds.min[1]) / (COLLIDER_SDF_RESOLUTION - 1);
        const dz = (bounds.max[2] - bounds.min[2]) / (COLLIDER_SDF_RESOLUTION - 1);
        const stack = new Uint32Array(nodes.length);

        for (let z = 0; z < COLLIDER_SDF_RESOLUTION; z++) {
            const pz = bounds.min[2] + z * dz;
            for (let y = 0; y < COLLIDER_SDF_RESOLUTION; y++) {
                const py = bounds.min[1] + y * dy;
                for (let x = 0; x < COLLIDER_SDF_RESOLUTION; x++) {
                    const px = bounds.min[0] + x * dx;
                    const index = x
                        + y * COLLIDER_SDF_RESOLUTION
                        + z * COLLIDER_SDF_RESOLUTION * COLLIDER_SDF_RESOLUTION;
                    const distSq = this.closestDistanceSqToMesh(
                        [px, py, pz],
                        triangles,
                        triangleIndices,
                        nodes,
                        stack,
                    );
                    sdfData[index] = Math.sqrt(distSq) - COLLIDER_SDF_SURFACE_THICKNESS;
                }
            }
        }

        return sdfData;
    }

    private buildSdfBvh(
        vertices: number[],
        indices: number[],
    ): { triangles: Triangle[], triangleIndices: number[], nodes: SdfBvhNode[] } {
        const numTriangles = indices.length / 3;

        if (numTriangles === 0) {
            return {
                triangles: [],
                triangleIndices: [],
                nodes: [],
            };
        }

        const triangles: Triangle[] = [];
        for (let i = 0; i < numTriangles; i++) {
            const idx0 = indices[i * 3 + 0];
            const idx1 = indices[i * 3 + 1];
            const idx2 = indices[i * 3 + 2];

            const v0: Vec3 = [vertices[idx0 * 3], vertices[idx0 * 3 + 1], vertices[idx0 * 3 + 2]];
            const v1: Vec3 = [vertices[idx1 * 3], vertices[idx1 * 3 + 1], vertices[idx1 * 3 + 2]];
            const v2: Vec3 = [vertices[idx2 * 3], vertices[idx2 * 3 + 1], vertices[idx2 * 3 + 2]];

            const minBounds: Vec3 = [
                Math.min(v0[0], v1[0], v2[0]),
                Math.min(v0[1], v1[1], v2[1]),
                Math.min(v0[2], v1[2], v2[2])
            ];
            const maxBounds: Vec3 = [
                Math.max(v0[0], v1[0], v2[0]),
                Math.max(v0[1], v1[1], v2[1]),
                Math.max(v0[2], v1[2], v2[2])
            ];
            const centroid: Vec3 = [
                (v0[0] + v1[0] + v2[0]) / 3,
                (v0[1] + v1[1] + v2[1]) / 3,
                (v0[2] + v1[2] + v2[2]) / 3
            ];

            triangles.push({ v0, v1, v2, centroid, minBounds, maxBounds });
        }

        const nodes: SdfBvhNode[] = [];
        const triangleIndices: number[] = Array.from({ length: numTriangles }, (_, i) => i);

        this.buildSdfBvhRecursive(triangles, triangleIndices, 0, numTriangles, nodes);

        return { triangles, triangleIndices, nodes };
    }

    private buildSdfBvhRecursive(
        triangles: Triangle[],
        triangleIndices: number[],
        start: number,
        end: number,
        nodes: SdfBvhNode[],
    ): number {
        const nodeIndex = nodes.length;
        const count = end - start;

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

        const MAX_LEAF_SIZE = 4;
        if (count <= MAX_LEAF_SIZE) {
            nodes.push({
                minBounds: [minX, minY, minZ],
                maxBounds: [maxX, maxY, maxZ],
                leftChild: 0,
                rightChild: 0,
                start,
                count,
                isLeaf: true,
            });
            return nodeIndex;
        }

        const extentX = maxX - minX;
        const extentY = maxY - minY;
        const extentZ = maxZ - minZ;

        let splitAxis = 0;
        if (extentY > extentX && extentY > extentZ) splitAxis = 1;
        else if (extentZ > extentX && extentZ > extentY) splitAxis = 2;

        const subIndices = triangleIndices.slice(start, end);
        subIndices.sort((a, b) => triangles[a].centroid[splitAxis] - triangles[b].centroid[splitAxis]);
        for (let i = 0; i < subIndices.length; i++) {
            triangleIndices[start + i] = subIndices[i];
        }

        const mid = start + Math.floor(count / 2);

        nodes.push({
            minBounds: [minX, minY, minZ],
            maxBounds: [maxX, maxY, maxZ],
            leftChild: 0,
            rightChild: 0,
            start,
            count,
            isLeaf: false,
        });

        const leftChild = this.buildSdfBvhRecursive(triangles, triangleIndices, start, mid, nodes);
        const rightChild = this.buildSdfBvhRecursive(triangles, triangleIndices, mid, end, nodes);

        nodes[nodeIndex].leftChild = leftChild;
        nodes[nodeIndex].rightChild = rightChild;

        return nodeIndex;
    }

    private closestDistanceSqToMesh(
        point: Vec3,
        triangles: Triangle[],
        triangleIndices: number[],
        nodes: SdfBvhNode[],
        stack: Uint32Array,
    ): number {
        let bestDistSq = Infinity;
        let stackPtr = 0;
        stack[stackPtr++] = 0;

        while (stackPtr > 0) {
            const node = nodes[stack[--stackPtr]];
            if (this.pointAabbDistanceSq(point, node.minBounds, node.maxBounds) > bestDistSq) {
                continue;
            }

            if (node.isLeaf) {
                const end = node.start + node.count;
                for (let i = node.start; i < end; i++) {
                    const tri = triangles[triangleIndices[i]];
                    const distSq = this.pointTriangleDistanceSq(point, tri);
                    if (distSq < bestDistSq) {
                        bestDistSq = distSq;
                    }
                }
                continue;
            }

            const left = nodes[node.leftChild];
            const right = nodes[node.rightChild];
            const leftDistSq = this.pointAabbDistanceSq(point, left.minBounds, left.maxBounds);
            const rightDistSq = this.pointAabbDistanceSq(point, right.minBounds, right.maxBounds);

            if (leftDistSq < rightDistSq) {
                if (rightDistSq <= bestDistSq) stack[stackPtr++] = node.rightChild;
                if (leftDistSq <= bestDistSq) stack[stackPtr++] = node.leftChild;
            } else {
                if (leftDistSq <= bestDistSq) stack[stackPtr++] = node.leftChild;
                if (rightDistSq <= bestDistSq) stack[stackPtr++] = node.rightChild;
            }
        }

        return bestDistSq;
    }

    private pointAabbDistanceSq(point: Vec3, minBounds: Vec3, maxBounds: Vec3): number {
        let distSq = 0;

        for (let axis = 0; axis < 3; axis++) {
            if (point[axis] < minBounds[axis]) {
                const d = minBounds[axis] - point[axis];
                distSq += d * d;
            } else if (point[axis] > maxBounds[axis]) {
                const d = point[axis] - maxBounds[axis];
                distSq += d * d;
            }
        }

        return distSq;
    }

    private pointTriangleDistanceSq(point: Vec3, tri: Triangle): number {
        const ax = tri.v0[0], ay = tri.v0[1], az = tri.v0[2];
        const bx = tri.v1[0], by = tri.v1[1], bz = tri.v1[2];
        const cx = tri.v2[0], cy = tri.v2[1], cz = tri.v2[2];
        const px = point[0], py = point[1], pz = point[2];

        const abx = bx - ax, aby = by - ay, abz = bz - az;
        const acx = cx - ax, acy = cy - ay, acz = cz - az;
        const apx = px - ax, apy = py - ay, apz = pz - az;

        const d1 = abx * apx + aby * apy + abz * apz;
        const d2 = acx * apx + acy * apy + acz * apz;
        if (d1 <= 0 && d2 <= 0) {
            return this.distanceSq(px, py, pz, ax, ay, az);
        }

        const bpx = px - bx, bpy = py - by, bpz = pz - bz;
        const d3 = abx * bpx + aby * bpy + abz * bpz;
        const d4 = acx * bpx + acy * bpy + acz * bpz;
        if (d3 >= 0 && d4 <= d3) {
            return this.distanceSq(px, py, pz, bx, by, bz);
        }

        const vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) {
            const v = d1 / (d1 - d3);
            return this.distanceSq(px, py, pz, ax + v * abx, ay + v * aby, az + v * abz);
        }

        const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
        const d5 = abx * cpx + aby * cpy + abz * cpz;
        const d6 = acx * cpx + acy * cpy + acz * cpz;
        if (d6 >= 0 && d5 <= d6) {
            return this.distanceSq(px, py, pz, cx, cy, cz);
        }

        const vb = d5 * d2 - d1 * d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0) {
            const w = d2 / (d2 - d6);
            return this.distanceSq(px, py, pz, ax + w * acx, ay + w * acy, az + w * acz);
        }

        const va = d3 * d6 - d5 * d4;
        if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
            const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return this.distanceSq(px, py, pz, bx + w * (cx - bx), by + w * (cy - by), bz + w * (cz - bz));
        }

        const denom = 1 / (va + vb + vc);
        const v = vb * denom;
        const w = vc * denom;
        return this.distanceSq(
            px,
            py,
            pz,
            ax + abx * v + acx * w,
            ay + aby * v + acy * w,
            az + abz * v + acz * w,
        );
    }

    private distanceSq(
        ax: number,
        ay: number,
        az: number,
        bx: number,
        by: number,
        bz: number,
    ): number {
        const dx = ax - bx;
        const dy = ay - by;
        const dz = az - bz;
        return dx * dx + dy * dy + dz * dz;
    }

    destroy() {
        this.colliderDataBuffer.destroy();
        this.colliderSdfBuffer.destroy();
        this.textureArray?.destroy();
    }
}
