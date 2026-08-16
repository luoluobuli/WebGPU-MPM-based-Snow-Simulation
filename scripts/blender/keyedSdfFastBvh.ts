import type { NearestTriangle, Vec3 } from "./keyedSdfBvh.ts";

type BvhNode = {
    minX: number;
    minY: number;
    minZ: number;
    maxX: number;
    maxY: number;
    maxZ: number;
    start: number;
    count: number;
    left: number;
    right: number;
};

const LEAF_TRIANGLE_COUNT = 8;
const RAY_EPSILON = 1e-8;
const RAY_DIRECTIONS: readonly Vec3[] = [
    [0.812147, 0.334879, 0.477293],
    [-0.271903, 0.924117, 0.267401],
    [0.394711, -0.519037, 0.758213],
];

const squaredDistanceToBox = (
    x: number,
    y: number,
    z: number,
    node: BvhNode,
) => {
    const dx = x < node.minX
        ? node.minX - x
        : x > node.maxX
        ? x - node.maxX
        : 0;
    const dy = y < node.minY
        ? node.minY - y
        : y > node.maxY
        ? y - node.maxY
        : 0;
    const dz = z < node.minZ
        ? node.minZ - z
        : z > node.maxZ
        ? z - node.maxZ
        : 0;

    return dx * dx + dy * dy + dz * dz;
};

export class FastTriangleBvh {
    readonly positions: Float32Array;
    readonly indices: Uint32Array;
    readonly triangleCount: number;

    private readonly triangleOrder: number[];
    private readonly triangleBounds: Float32Array;
    private readonly triangleCentroids: Float32Array;
    private readonly nodes: BvhNode[] = [];
    private readonly stack: number[] = [];
    private readonly nearestResult: NearestTriangle = {
        squaredDistance: Infinity,
        triangleIndex: -1,
        barycentric: [0, 0, 0],
    };

    constructor(positions: Float32Array, indices: Uint32Array) {
        if (indices.length === 0 || indices.length % 3 !== 0) {
            throw new Error("BVH indices must contain complete triangles");
        }

        this.positions = positions;
        this.indices = indices;
        this.triangleCount = indices.length / 3;
        this.triangleOrder = Array.from(
            { length: this.triangleCount },
            (_, triangleIndex) => triangleIndex,
        );
        this.triangleBounds = new Float32Array(
            this.triangleCount * 6,
        );
        this.triangleCentroids = new Float32Array(
            this.triangleCount * 3,
        );
        this.precomputeTriangles();
        this.buildNode(0, this.triangleCount);
    }

    private coordinate(vertexIndex: number, axis: number) {
        return this.positions[vertexIndex * 3 + axis];
    }

    private precomputeTriangles() {
        for (
            let triangleIndex = 0;
            triangleIndex < this.triangleCount;
            triangleIndex++
        ) {
            const indexOffset = triangleIndex * 3;
            const first = this.indices[indexOffset];
            const second = this.indices[indexOffset + 1];
            const third = this.indices[indexOffset + 2];
            const boundsOffset = triangleIndex * 6;
            const centroidOffset = triangleIndex * 3;

            for (let axis = 0; axis < 3; axis++) {
                const a = this.coordinate(first, axis);
                const b = this.coordinate(second, axis);
                const c = this.coordinate(third, axis);
                this.triangleBounds[boundsOffset + axis] = Math.min(
                    a,
                    b,
                    c,
                );
                this.triangleBounds[boundsOffset + 3 + axis] = Math.max(
                    a,
                    b,
                    c,
                );
                this.triangleCentroids[centroidOffset + axis] = (a + b + c) / 3;
            }
        }
    }

    private buildNode(start: number, count: number): number {
        let minX = Infinity;
        let minY = Infinity;
        let minZ = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;
        let maxZ = -Infinity;
        let centroidMinX = Infinity;
        let centroidMinY = Infinity;
        let centroidMinZ = Infinity;
        let centroidMaxX = -Infinity;
        let centroidMaxY = -Infinity;
        let centroidMaxZ = -Infinity;

        for (let offset = start; offset < start + count; offset++) {
            const triangleIndex = this.triangleOrder[offset];
            const boundsOffset = triangleIndex * 6;
            const centroidOffset = triangleIndex * 3;
            minX = Math.min(minX, this.triangleBounds[boundsOffset]);
            minY = Math.min(
                minY,
                this.triangleBounds[boundsOffset + 1],
            );
            minZ = Math.min(
                minZ,
                this.triangleBounds[boundsOffset + 2],
            );
            maxX = Math.max(
                maxX,
                this.triangleBounds[boundsOffset + 3],
            );
            maxY = Math.max(
                maxY,
                this.triangleBounds[boundsOffset + 4],
            );
            maxZ = Math.max(
                maxZ,
                this.triangleBounds[boundsOffset + 5],
            );
            centroidMinX = Math.min(
                centroidMinX,
                this.triangleCentroids[centroidOffset],
            );
            centroidMinY = Math.min(
                centroidMinY,
                this.triangleCentroids[centroidOffset + 1],
            );
            centroidMinZ = Math.min(
                centroidMinZ,
                this.triangleCentroids[centroidOffset + 2],
            );
            centroidMaxX = Math.max(
                centroidMaxX,
                this.triangleCentroids[centroidOffset],
            );
            centroidMaxY = Math.max(
                centroidMaxY,
                this.triangleCentroids[centroidOffset + 1],
            );
            centroidMaxZ = Math.max(
                centroidMaxZ,
                this.triangleCentroids[centroidOffset + 2],
            );
        }

        const nodeIndex = this.nodes.length;
        this.nodes.push({
            minX,
            minY,
            minZ,
            maxX,
            maxY,
            maxZ,
            start,
            count,
            left: -1,
            right: -1,
        });
        if (count <= LEAF_TRIANGLE_COUNT) return nodeIndex;

        const extentX = centroidMaxX - centroidMinX;
        const extentY = centroidMaxY - centroidMinY;
        const extentZ = centroidMaxZ - centroidMinZ;
        const axis = extentY > extentX
            ? extentZ > extentY ? 2 : 1
            : extentZ > extentX
            ? 2
            : 0;
        const sorted = this.triangleOrder
            .slice(start, start + count)
            .sort(
                (first, second) =>
                    this.triangleCentroids[first * 3 + axis] -
                    this.triangleCentroids[second * 3 + axis],
            );
        for (let offset = 0; offset < count; offset++) {
            this.triangleOrder[start + offset] = sorted[offset];
        }

        const leftCount = Math.floor(count / 2);
        this.nodes[nodeIndex].left = this.buildNode(start, leftCount);
        this.nodes[nodeIndex].right = this.buildNode(
            start + leftCount,
            count - leftCount,
        );
        return nodeIndex;
    }

    private testTriangle(
        triangleIndex: number,
        px: number,
        py: number,
        pz: number,
        result: NearestTriangle,
    ) {
        const indexOffset = triangleIndex * 3;
        const ai = this.indices[indexOffset] * 3;
        const bi = this.indices[indexOffset + 1] * 3;
        const ci = this.indices[indexOffset + 2] * 3;
        const ax = this.positions[ai];
        const ay = this.positions[ai + 1];
        const az = this.positions[ai + 2];
        const abx = this.positions[bi] - ax;
        const aby = this.positions[bi + 1] - ay;
        const abz = this.positions[bi + 2] - az;
        const acx = this.positions[ci] - ax;
        const acy = this.positions[ci + 1] - ay;
        const acz = this.positions[ci + 2] - az;
        const apx = px - ax;
        const apy = py - ay;
        const apz = pz - az;
        const d1 = abx * apx + aby * apy + abz * apz;
        const d2 = acx * apx + acy * apy + acz * apz;
        let u = 1;
        let v = 0;
        let w = 0;

        if (!(d1 <= 0 && d2 <= 0)) {
            const bx = this.positions[bi];
            const by = this.positions[bi + 1];
            const bz = this.positions[bi + 2];
            const bpx = px - bx;
            const bpy = py - by;
            const bpz = pz - bz;
            const d3 = abx * bpx + aby * bpy + abz * bpz;
            const d4 = acx * bpx + acy * bpy + acz * bpz;
            if (d3 >= 0 && d4 <= d3) {
                u = 0;
                v = 1;
            } else {
                const vc = d1 * d4 - d3 * d2;
                if (vc <= 0 && d1 >= 0 && d3 <= 0) {
                    v = d1 / (d1 - d3);
                    u = 1 - v;
                } else {
                    const cx = this.positions[ci];
                    const cy = this.positions[ci + 1];
                    const cz = this.positions[ci + 2];
                    const cpx = px - cx;
                    const cpy = py - cy;
                    const cpz = pz - cz;
                    const d5 = abx * cpx + aby * cpy + abz * cpz;
                    const d6 = acx * cpx + acy * cpy + acz * cpz;
                    if (d6 >= 0 && d5 <= d6) {
                        u = 0;
                        w = 1;
                    } else {
                        const vb = d5 * d2 - d1 * d6;
                        if (vb <= 0 && d2 >= 0 && d6 <= 0) {
                            w = d2 / (d2 - d6);
                            u = 1 - w;
                        } else {
                            const va = d3 * d6 - d5 * d4;
                            if (
                                va <= 0 &&
                                d4 - d3 >= 0 &&
                                d5 - d6 >= 0
                            ) {
                                w = (d4 - d3) / (
                                    d4 - d3 + d5 - d6
                                );
                                v = 1 - w;
                                u = 0;
                            } else {
                                const inverse = 1 / (va + vb + vc);
                                v = vb * inverse;
                                w = vc * inverse;
                                u = 1 - v - w;
                            }
                        }
                    }
                }
            }
        }

        const dx = px - (
            ax * u + this.positions[bi] * v +
            this.positions[ci] * w
        );
        const dy = py - (
            ay * u + this.positions[bi + 1] * v +
            this.positions[ci + 1] * w
        );
        const dz = pz - (
            az * u + this.positions[bi + 2] * v +
            this.positions[ci + 2] * w
        );
        const squaredDistance = dx * dx + dy * dy + dz * dz;
        if (squaredDistance < result.squaredDistance) {
            result.squaredDistance = squaredDistance;
            result.triangleIndex = triangleIndex;
            result.barycentric[0] = u;
            result.barycentric[1] = v;
            result.barycentric[2] = w;
        }
    }

    nearest(point: Vec3) {
        const result = this.nearestResult;
        result.squaredDistance = Infinity;
        result.triangleIndex = -1;
        const stack = this.stack;
        stack.length = 1;
        stack[0] = 0;

        while (stack.length > 0) {
            const node = this.nodes[stack.pop()!];
            if (
                squaredDistanceToBox(
                    point[0],
                    point[1],
                    point[2],
                    node,
                ) >= result.squaredDistance
            ) {
                continue;
            }
            if (node.left >= 0) {
                stack.push(node.left, node.right);
                continue;
            }

            for (
                let offset = node.start;
                offset < node.start + node.count;
                offset++
            ) {
                this.testTriangle(
                    this.triangleOrder[offset],
                    point[0],
                    point[1],
                    point[2],
                    result,
                );
            }
        }

        return result;
    }
    private rayIntersectsBox(
        origin: Vec3,
        direction: Vec3,
        node: BvhNode,
    ) {
        let near = 0;
        let far = Infinity;

        let inverseDirection = 1 / direction[0];
        let axisNear = (node.minX - origin[0]) * inverseDirection;
        let axisFar = (node.maxX - origin[0]) * inverseDirection;
        if (axisNear > axisFar) {
            const swap = axisNear;
            axisNear = axisFar;
            axisFar = swap;
        }
        near = Math.max(near, axisNear);
        far = Math.min(far, axisFar);
        if (near > far) return false;

        inverseDirection = 1 / direction[1];
        axisNear = (node.minY - origin[1]) * inverseDirection;
        axisFar = (node.maxY - origin[1]) * inverseDirection;
        if (axisNear > axisFar) {
            const swap = axisNear;
            axisNear = axisFar;
            axisFar = swap;
        }
        near = Math.max(near, axisNear);
        far = Math.min(far, axisFar);
        if (near > far) return false;

        inverseDirection = 1 / direction[2];
        axisNear = (node.minZ - origin[2]) * inverseDirection;
        axisFar = (node.maxZ - origin[2]) * inverseDirection;
        if (axisNear > axisFar) {
            const swap = axisNear;
            axisNear = axisFar;
            axisFar = swap;
        }
        near = Math.max(near, axisNear);
        far = Math.min(far, axisFar);

        return far > RAY_EPSILON;
    }

    private rayIntersectsTriangle(
        origin: Vec3,
        direction: Vec3,
        triangleIndex: number,
    ) {
        const indexOffset = triangleIndex * 3;
        const ai = this.indices[indexOffset] * 3;
        const bi = this.indices[indexOffset + 1] * 3;
        const ci = this.indices[indexOffset + 2] * 3;
        const ax = this.positions[ai];
        const ay = this.positions[ai + 1];
        const az = this.positions[ai + 2];
        const edge1X = this.positions[bi] - ax;
        const edge1Y = this.positions[bi + 1] - ay;
        const edge1Z = this.positions[bi + 2] - az;
        const edge2X = this.positions[ci] - ax;
        const edge2Y = this.positions[ci + 1] - ay;
        const edge2Z = this.positions[ci + 2] - az;
        const pX = direction[1] * edge2Z - direction[2] * edge2Y;
        const pY = direction[2] * edge2X - direction[0] * edge2Z;
        const pZ = direction[0] * edge2Y - direction[1] * edge2X;
        const determinant = edge1X * pX + edge1Y * pY + edge1Z * pZ;
        if (Math.abs(determinant) <= RAY_EPSILON) return false;

        const inverseDeterminant = 1 / determinant;
        const tX = origin[0] - ax;
        const tY = origin[1] - ay;
        const tZ = origin[2] - az;
        const u = (tX * pX + tY * pY + tZ * pZ) *
            inverseDeterminant;
        if (u < -RAY_EPSILON || u > 1 + RAY_EPSILON) return false;

        const qX = tY * edge1Z - tZ * edge1Y;
        const qY = tZ * edge1X - tX * edge1Z;
        const qZ = tX * edge1Y - tY * edge1X;
        const v = (
            direction[0] * qX +
            direction[1] * qY +
            direction[2] * qZ
        ) * inverseDeterminant;
        if (
            v < -RAY_EPSILON ||
            u + v > 1 + RAY_EPSILON
        ) {
            return false;
        }

        const distance = (
            edge2X * qX + edge2Y * qY + edge2Z * qZ
        ) * inverseDeterminant;
        return distance > RAY_EPSILON;
    }

    contains(point: Vec3) {
        let insideVotes = 0;
        for (const direction of RAY_DIRECTIONS) {
            let intersections = 0;
            const stack = this.stack;
            stack.length = 1;
            stack[0] = 0;
            while (stack.length > 0) {
                const node = this.nodes[stack.pop()!];
                if (!this.rayIntersectsBox(point, direction, node)) {
                    continue;
                }
                if (node.left >= 0) {
                    stack.push(node.left, node.right);
                    continue;
                }
                for (
                    let offset = node.start;
                    offset < node.start + node.count;
                    offset++
                ) {
                    if (
                        this.rayIntersectsTriangle(
                            point,
                            direction,
                            this.triangleOrder[offset],
                        )
                    ) {
                        intersections++;
                    }
                }
            }
            insideVotes += intersections % 2;
        }
        return insideVotes >= 2;
    }
}
