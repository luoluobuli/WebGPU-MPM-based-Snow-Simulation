import type { NearestTriangle, Vec3 } from "./keyedSdfBvh.ts";
import { FastTriangleBvh } from "./keyedSdfFastBvh.ts";
import {
    applyPawMeshClosure,
    type PawMeshClosure,
} from "./zubeiaPawClosure.ts";
import type {
    MeshFrame,
    PackedPawField,
    PawSource,
    SdfGridDimensions,
} from "./zubeiaKeyedSdfTypes.ts";

export type FrameBvhs = Map<string, FastTriangleBvh>;

type ClosestSurface = {
    meshName: string;
    nearest: NearestTriangle;
};

const voxelIndex = (
    x: number,
    y: number,
    z: number,
    dimensions: SdfGridDimensions,
) => x + y * dimensions[0] + z * dimensions[0] * dimensions[1];

const framePosition = (
    frame: MeshFrame,
    vertexIndex: number,
): Vec3 => {
    const offset = vertexIndex * 3;

    return [
        frame.positions[offset],
        frame.positions[offset + 1],
        frame.positions[offset + 2],
    ];
};

const closestPoint = (
    frame: MeshFrame,
    nearest: NearestTriangle,
) => {
    const indexOffset = nearest.triangleIndex * 3;
    const a = framePosition(frame, frame.indices[indexOffset]);
    const b = framePosition(frame, frame.indices[indexOffset + 1]);
    const c = framePosition(frame, frame.indices[indexOffset + 2]);
    const barycentric = nearest.barycentric;

    return [
        a[0] * barycentric[0] +
        b[0] * barycentric[1] +
        c[0] * barycentric[2],
        a[1] * barycentric[0] +
        b[1] * barycentric[1] +
        c[1] * barycentric[2],
        a[2] * barycentric[0] +
        b[2] * barycentric[1] +
        c[2] * barycentric[2],
    ] as Vec3;
};

const surfaceVelocity = ({
    previousFrame,
    nextFrame,
    nearest,
    inverseDeltaS,
}: {
    previousFrame: MeshFrame;
    nextFrame: MeshFrame;
    nearest: NearestTriangle;
    inverseDeltaS: number;
}) => {
    const previous = closestPoint(previousFrame, nearest);
    const next = closestPoint(nextFrame, nearest);

    return [
        (next[0] - previous[0]) * inverseDeltaS,
        (next[1] - previous[1]) * inverseDeltaS,
        (next[2] - previous[2]) * inverseDeltaS,
    ] as Vec3;
};

const findClosestSurface = ({
    point,
    meshNames,
    bvhs,
}: {
    point: Vec3;
    meshNames: readonly string[];
    bvhs: ReadonlyMap<string, FastTriangleBvh>;
}) => {
    let result: ClosestSurface | null = null;

    meshNames.forEach((meshName) => {
        const bvh = bvhs.get(meshName)!;
        const nearest = bvh.nearest(point);
        if (
            result === null ||
            nearest.squaredDistance <
                result.nearest.squaredDistance
        ) {
            result = {
                meshName,
                nearest,
            };
        }
    });

    return result!;
};

const pointIsInsideAnyBvh = (
    point: Vec3,
    bvhs: ReadonlyMap<string, FastTriangleBvh>,
) => {
    for (const bvh of bvhs.values()) {
        if (bvh.contains(point)) return true;
    }
    return false;
};

export const createFrameBvhs = (
    frame: ReadonlyMap<string, MeshFrame>,
    meshClosures: ReadonlyMap<string, PawMeshClosure>,
) => {
    const result: FrameBvhs = new Map();
    meshClosures.forEach((closure, meshName) => {
        const closedFrame = applyPawMeshClosure({
            frame: frame.get(meshName)!,
            closure,
        });
        result.set(
            meshName,
            new FastTriangleBvh(
                closedFrame.positions,
                closedFrame.indices,
            ),
        );
    });

    return result;
};

export const bakePawField = ({
    paw,
    resolution,
    previousFrame,
    nextFrame,
    currentBvhs,
    sampleRateHz,
    gridDimensions,
}: {
    paw: PawSource;
    resolution: number;
    previousFrame: ReadonlyMap<string, MeshFrame>;
    nextFrame: ReadonlyMap<string, MeshFrame>;
    currentBvhs: ReadonlyMap<string, FastTriangleBvh>;
    sampleRateHz: number;
    gridDimensions?: SdfGridDimensions;
}) => {
    const dimensions = gridDimensions ?? [
        resolution,
        resolution,
        resolution,
    ] satisfies SdfGridDimensions;
    if (
        dimensions.some(
            value => !Number.isInteger(value) || value < 4,
        )
    ) {
        throw new Error("SDF grid dimensions must be integers >= 4");
    }
    const voxelCount =
        dimensions[0] * dimensions[1] * dimensions[2];
    const phi = new Float32Array(voxelCount);
    const velocity = new Float32Array(voxelCount * 3);
    const min = paw.bounds.minLocal;
    const max = paw.bounds.maxLocal;
    const cellSize: Vec3 = [
        (max[0] - min[0]) / (dimensions[0] - 1),
        (max[1] - min[1]) / (dimensions[1] - 1),
        (max[2] - min[2]) / (dimensions[2] - 1),
    ];
    const sealingDistanceM = 0;
    const inverseVelocityDeltaS = sampleRateHz * 0.5;
    let negativeVoxelCount = 0;
    const previousClosedFrames = new Map<string, MeshFrame>();
    const nextClosedFrames = new Map<string, MeshFrame>();
    paw.meshClosures.forEach((closure, meshName) => {
        previousClosedFrames.set(
            meshName,
            applyPawMeshClosure({
                frame: previousFrame.get(meshName)!,
                closure,
            }),
        );
        nextClosedFrames.set(
            meshName,
            applyPawMeshClosure({
                frame: nextFrame.get(meshName)!,
                closure,
            }),
        );
    });

    for (let z = 0; z < dimensions[2]; z++) {
        for (let y = 0; y < dimensions[1]; y++) {
            for (let x = 0; x < dimensions[0]; x++) {
                const index = voxelIndex(x, y, z, dimensions);
                const point: Vec3 = [
                    min[0] + x * cellSize[0],
                    min[1] + y * cellSize[1],
                    min[2] + z * cellSize[2],
                ];
                const closest = findClosestSurface({
                    point,
                    meshNames: paw.manifest.solid_meshes,
                    bvhs: currentBvhs,
                });
                const distance = Math.sqrt(
                    closest.nearest.squaredDistance,
                );
                const velocityValue = surfaceVelocity({
                    previousFrame: previousClosedFrames.get(
                        closest.meshName,
                    )!,
                    nextFrame: nextClosedFrames.get(
                        closest.meshName,
                    )!,
                    nearest: closest.nearest,
                    inverseDeltaS: inverseVelocityDeltaS,
                });

                const isInside = pointIsInsideAnyBvh(
                    point,
                    currentBvhs,
                );
                phi[index] = isInside ? -distance : distance;
                negativeVoxelCount += Number(isInside);
                const velocityOffset = index * 3;
                velocity[velocityOffset] = velocityValue[0];
                velocity[velocityOffset + 1] = velocityValue[1];
                velocity[velocityOffset + 2] = velocityValue[2];
            }
        }
    }

    return {
        field: {
            phi,
            velocity,
        } satisfies PackedPawField,
        cellSize,
        sealingDistanceM,
        negativeVoxelCount,
    };
};

const floatBits = new DataView(new ArrayBuffer(4));

export const float32ToFloat16Bits = (value: number) => {
    floatBits.setFloat32(0, value, false);
    const bits = floatBits.getUint32(0, false);
    const sign = (bits >>> 16) & 0x8000;
    const exponent = (bits >>> 23) & 0xff;
    const mantissa = bits & 0x7fffff;

    if (exponent === 0xff) {
        return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
    }

    let halfExponent = exponent - 127 + 15;
    if (halfExponent >= 0x1f) return sign | 0x7c00;
    if (halfExponent <= 0) {
        if (halfExponent < -10) return sign;
        const hiddenMantissa = mantissa | 0x800000;
        const shift = 14 - halfExponent;
        const rounded = (
            hiddenMantissa + (1 << (shift - 1)) - 1 +
            ((hiddenMantissa >> shift) & 1)
        ) >> shift;
        return sign | rounded;
    }

    const roundedMantissa = mantissa + 0xfff +
        ((mantissa >> 13) & 1);
    if ((roundedMantissa & 0x800000) !== 0) {
        halfExponent++;
        if (halfExponent >= 0x1f) return sign | 0x7c00;
        return sign | (halfExponent << 10);
    }

    return sign | (halfExponent << 10) |
        (roundedMantissa >> 13);
};
