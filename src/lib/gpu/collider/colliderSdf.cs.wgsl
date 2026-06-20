const COLLIDER_SDF_CREATION_WORKGROUP_SIZE = 64u;
const COLLIDER_SDF_CREATION_SURFACE_THICKNESS = 0.05;
const COLLIDER_SDF_CREATION_MAX_FLOAT = 3.402823e38;
const COLLIDER_SDF_BVH_STACK_SIZE = 128u;

struct ColliderSdfCreationParams {
    minCoords: vec3f,
    resolution: u32,
    cellSize: vec3f,
    numTriangles: u32,
}

struct SdfBvhNode {
    minBounds: vec3f,
    leftChild: u32,
    maxBounds: vec3f,
    rightChild: u32,
    start: u32,
    count: u32,
    isLeaf: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> colliderIndices: array<u32>;
@group(0) @binding(1) var<storage, read> colliderVertices: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> colliderSdf: array<f32>;
@group(0) @binding(3) var<uniform> sdfParams: ColliderSdfCreationParams;
@group(0) @binding(4) var<storage, read> sdfBvhNodes: array<SdfBvhNode>;
@group(0) @binding(5) var<storage, read> sdfBvhTriangleOrder: array<u32>;

fn distanceSq(a: vec3f, b: vec3f) -> f32 {
    let delta = a - b;
    return dot(delta, delta);
}

fn pointAabbDistanceSq(point: vec3f, minBounds: vec3f, maxBounds: vec3f) -> f32 {
    let clampedPoint = clamp(point, minBounds, maxBounds);
    return distanceSq(point, clampedPoint);
}

fn pointSegmentDistanceSq(point: vec3f, a: vec3f, b: vec3f) -> f32 {
    let ab = b - a;
    let abLengthSq = dot(ab, ab);
    if abLengthSq <= 1e-20 {
        return distanceSq(point, a);
    }

    let t = clamp(dot(point - a, ab) / abLengthSq, 0.0, 1.0);
    return distanceSq(point, a + t * ab);
}

fn pointTriangleDistanceSq(point: vec3f, a: vec3f, b: vec3f, c: vec3f) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let triangleNormal = cross(ab, ac);
    if dot(triangleNormal, triangleNormal) <= 1e-20 {
        return min(
            pointSegmentDistanceSq(point, a, b),
            min(
                pointSegmentDistanceSq(point, b, c),
                pointSegmentDistanceSq(point, c, a),
            ),
        );
    }

    let ap = point - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return distanceSq(point, a);
    }

    let bp = point - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return distanceSq(point, b);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return distanceSq(point, a + v * ab);
    }

    let cp = point - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return distanceSq(point, c);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return distanceSq(point, a + w * ac);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return distanceSq(point, b + w * (c - b));
    }

    let denom = 1.0 / max(va + vb + vc, 1e-20);
    let v = vb * denom;
    let w = vc * denom;
    return distanceSq(point, a + ab * v + ac * w);
}

fn pointTriangleDistanceSqByTriangleIndex(point: vec3f, triangleIndex: u32) -> f32 {
    let indexOffset = triangleIndex * 3u;
    let v0 = colliderVertices[colliderIndices[indexOffset]].xyz;
    let v1 = colliderVertices[colliderIndices[indexOffset + 1u]].xyz;
    let v2 = colliderVertices[colliderIndices[indexOffset + 2u]].xyz;

    return pointTriangleDistanceSq(point, v0, v1, v2);
}

fn closestDistanceSqToMesh(point: vec3f) -> f32 {
    var bestDistanceSq = COLLIDER_SDF_CREATION_MAX_FLOAT;
    var stack: array<u32, 128>;
    var stackPtr = 1u;
    stack[0] = 0u;

    while stackPtr > 0u {
        stackPtr -= 1u;
        let node = sdfBvhNodes[stack[stackPtr]];
        if pointAabbDistanceSq(point, node.minBounds, node.maxBounds) > bestDistanceSq {
            continue;
        }

        if node.isLeaf != 0u {
            for (var i = 0u; i < node.count; i++) {
                let triangleIndex = sdfBvhTriangleOrder[node.start + i];
                bestDistanceSq = min(
                    bestDistanceSq,
                    pointTriangleDistanceSqByTriangleIndex(point, triangleIndex),
                );
            }
            continue;
        }

        let leftNode = sdfBvhNodes[node.leftChild];
        let rightNode = sdfBvhNodes[node.rightChild];
        let leftDistanceSq = pointAabbDistanceSq(point, leftNode.minBounds, leftNode.maxBounds);
        let rightDistanceSq = pointAabbDistanceSq(point, rightNode.minBounds, rightNode.maxBounds);

        if leftDistanceSq < rightDistanceSq {
            if rightDistanceSq <= bestDistanceSq && stackPtr < COLLIDER_SDF_BVH_STACK_SIZE {
                stack[stackPtr] = node.rightChild;
                stackPtr += 1u;
            }
            if leftDistanceSq <= bestDistanceSq && stackPtr < COLLIDER_SDF_BVH_STACK_SIZE {
                stack[stackPtr] = node.leftChild;
                stackPtr += 1u;
            }
        } else {
            if leftDistanceSq <= bestDistanceSq && stackPtr < COLLIDER_SDF_BVH_STACK_SIZE {
                stack[stackPtr] = node.leftChild;
                stackPtr += 1u;
            }
            if rightDistanceSq <= bestDistanceSq && stackPtr < COLLIDER_SDF_BVH_STACK_SIZE {
                stack[stackPtr] = node.rightChild;
                stackPtr += 1u;
            }
        }
    }

    return bestDistanceSq;
}

@compute
@workgroup_size(64)
fn createColliderSdf(@builtin(global_invocation_id) globalId: vec3u) {
    let voxelIndex = globalId.x;
    let resolution = sdfParams.resolution;
    let sliceVoxelCount = resolution * resolution;
    let voxelCount = sliceVoxelCount * resolution;
    if voxelIndex >= voxelCount {
        return;
    }

    let x = voxelIndex % resolution;
    let y = (voxelIndex / resolution) % resolution;
    let z = voxelIndex / sliceVoxelCount;
    let point = sdfParams.minCoords
        + vec3f(f32(x), f32(y), f32(z)) * sdfParams.cellSize;

    colliderSdf[voxelIndex] = sqrt(closestDistanceSqToMesh(point))
        - COLLIDER_SDF_CREATION_SURFACE_THICKNESS;
}