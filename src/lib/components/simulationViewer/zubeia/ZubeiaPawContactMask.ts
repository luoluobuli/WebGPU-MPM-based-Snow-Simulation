import { mat4, type Mat4 } from "wgpu-matrix";

import type {
    RigidPawSdfKinematicColliderAnimation,
} from "$lib/gpu/kinematicCollider/RigidPawSdfKinematicCollider";
import type { ZubeiaGroundBounds } from "./ZubeiaGround";

export type ZubeiaPawContactMask = {
    walkingCellCount: number,
    crossStripCellCount: number,
    cellSizeM: number,
    contact: Uint8Array,
    dilated: Uint8Array,
    perimeter: Uint8Array,
};

export type BuildZubeiaPawContactMaskOptions = {
    animation: RigidPawSdfKinematicColliderAnimation,
    groundStripBounds: ZubeiaGroundBounds,
    cellSizeM: number,
    groundSurfaceZM: number,
    outputFrameIndex: number,
    secondsPerFrame: number,
};

export const ZUBEIA_PAW_CONTACT_DILATION_CELLS = 2;

const MATRIX_FLOAT_COUNT = 16;

const float16BitsToNumber = (bits: number) => {
    const sign = (bits & 0x8000) === 0 ? 1 : -1;
    const exponent = (bits >>> 10) & 0x1f;
    const mantissa = bits & 0x3ff;

    if (exponent === 0) {
        return sign * 2 ** -14 * (mantissa / 2 ** 10);
    }
    if (exponent === 0x1f) {
        return mantissa === 0 ? sign * Infinity : Number.NaN;
    }

    return sign * 2 ** (exponent - 15) * (1 + mantissa / 2 ** 10);
};

const unpackSdf = (
    animation: RigidPawSdfKinematicColliderAnimation,
) => {
    const voxelCount =
        animation.primitiveCount * animation.sdfResolution ** 3;
    const expectedWordCount = Math.ceil(voxelCount / 2);
    if (animation.packedPhi.length !== expectedWordCount) {
        throw new Error(
            `rigid paw SDF must contain ${expectedWordCount} packed words`,
        );
    }

    const phi = new Float32Array(voxelCount);
    for (let voxelIndex = 0; voxelIndex < voxelCount; voxelIndex++) {
        const word = animation.packedPhi[voxelIndex >>> 1];
        const bits = (voxelIndex & 1) === 0
            ? word & 0xffff
            : word >>> 16;
        phi[voxelIndex] = float16BitsToNumber(bits);
    }

    return phi;
};

const sampleSdf = ({
    phi,
    resolution,
    primitiveIndex,
    unitX,
    unitY,
    unitZ,
}: {
    phi: Float32Array,
    resolution: number,
    primitiveIndex: number,
    unitX: number,
    unitY: number,
    unitZ: number,
}) => {
    if (
        unitX < -1
        || unitX > 1
        || unitY < -1
        || unitY > 1
        || unitZ < -1
        || unitZ > 1
    ) {
        return Infinity;
    }

    const lastGridIndex = resolution - 1;
    const maximumBaseIndex = resolution - 2;
    const gridX = (unitX * 0.5 + 0.5) * lastGridIndex;
    const gridY = (unitY * 0.5 + 0.5) * lastGridIndex;
    const gridZ = (unitZ * 0.5 + 0.5) * lastGridIndex;
    const baseX = Math.min(Math.floor(gridX), maximumBaseIndex);
    const baseY = Math.min(Math.floor(gridY), maximumBaseIndex);
    const baseZ = Math.min(Math.floor(gridZ), maximumBaseIndex);
    const interpolationX = gridX - baseX;
    const interpolationY = gridY - baseY;
    const interpolationZ = gridZ - baseZ;
    const layerStride = resolution * resolution;
    const fieldOffset = primitiveIndex * resolution ** 3;
    const valueAt = (x: number, y: number, z: number) => phi[
        fieldOffset + x + y * resolution + z * layerStride
    ];
    const mix = (first: number, second: number, alpha: number) =>
        first * (1 - alpha) + second * alpha;
    const x00 = mix(
        valueAt(baseX, baseY, baseZ),
        valueAt(baseX + 1, baseY, baseZ),
        interpolationX,
    );
    const x10 = mix(
        valueAt(baseX, baseY + 1, baseZ),
        valueAt(baseX + 1, baseY + 1, baseZ),
        interpolationX,
    );
    const x01 = mix(
        valueAt(baseX, baseY, baseZ + 1),
        valueAt(baseX + 1, baseY, baseZ + 1),
        interpolationX,
    );
    const x11 = mix(
        valueAt(baseX, baseY + 1, baseZ + 1),
        valueAt(baseX + 1, baseY + 1, baseZ + 1),
        interpolationX,
    );

    return mix(
        mix(x00, x10, interpolationY),
        mix(x01, x11, interpolationY),
        interpolationZ,
    );
};

const cellCountForExtent = (
    minimum: number,
    maximum: number,
    cellSizeM: number,
) => {
    const cellCount = Math.round((maximum - minimum) / cellSizeM);
    if (
        cellCount <= 0
        || Math.abs(minimum + cellCount * cellSizeM - maximum)
            > cellSizeM * 1e-5
    ) {
        throw new Error("ground strip extent must contain whole cells");
    }

    return cellCount;
};

const validateOptions = ({
    animation,
    groundStripBounds,
    cellSizeM,
    groundSurfaceZM,
    outputFrameIndex,
    secondsPerFrame,
}: BuildZubeiaPawContactMaskOptions) => {
    if (
        !Number.isInteger(animation.primitiveCount)
        || animation.primitiveCount <= 0
        || !Number.isInteger(animation.sdfResolution)
        || animation.sdfResolution < 2
    ) {
        throw new Error("rigid paw SDF dimensions must be positive integers");
    }
    if (!Number.isFinite(cellSizeM) || cellSizeM <= 0) {
        throw new Error("ground cell size must be positive");
    }
    if (!Number.isFinite(groundSurfaceZM)) {
        throw new Error("ground surface must be finite");
    }
    if (!Number.isInteger(outputFrameIndex) || outputFrameIndex < 0) {
        throw new Error("output frame index must be a non-negative integer");
    }
    if (!Number.isFinite(secondsPerFrame) || secondsPerFrame <= 0) {
        throw new Error("seconds per frame must be positive");
    }
    for (let axis = 0; axis < 3; axis++) {
        const minimum = groundStripBounds.min[axis];
        const maximum = groundStripBounds.max[axis];
        if (
            !Number.isFinite(minimum)
            || !Number.isFinite(maximum)
            || maximum <= minimum
        ) {
            throw new Error("ground strip bounds must be finite and ordered");
        }
    }
};

const dilateContact = ({
    contact,
    crossStripCellCount,
    walkingCellCount,
}: {
    contact: Uint8Array,
    crossStripCellCount: number,
    walkingCellCount: number,
}) => {
    const dilated = new Uint8Array(contact.length);

    // Two repetitions of 8-neighbor dilation form a two-cell Chebyshev ring.
    for (let x = 0; x < crossStripCellCount; x++) {
        for (let y = 0; y < walkingCellCount; y++) {
            const contactIndex = x * walkingCellCount + y;
            if (contact[contactIndex] === 0) continue;

            for (
                let offsetX = -ZUBEIA_PAW_CONTACT_DILATION_CELLS;
                offsetX <= ZUBEIA_PAW_CONTACT_DILATION_CELLS;
                offsetX++
            ) {
                const neighborX = x + offsetX;
                if (
                    neighborX < 0
                    || neighborX >= crossStripCellCount
                ) {
                    continue;
                }

                for (
                    let offsetY = -ZUBEIA_PAW_CONTACT_DILATION_CELLS;
                    offsetY <= ZUBEIA_PAW_CONTACT_DILATION_CELLS;
                    offsetY++
                ) {
                    const neighborY = y + offsetY;
                    if (
                        neighborY < 0
                        || neighborY >= walkingCellCount
                    ) {
                        continue;
                    }
                    dilated[
                        neighborX * walkingCellCount + neighborY
                    ] = 1;
                }
            }
        }
    }

    const perimeter = new Uint8Array(contact.length);
    for (let index = 0; index < contact.length; index++) {
        perimeter[index] = dilated[index] & (1 - contact[index]);
    }

    return { dilated, perimeter };
};

export const buildZubeiaPawContactMask = (
    options: BuildZubeiaPawContactMaskOptions,
): ZubeiaPawContactMask => {
    validateOptions(options);

    const {
        animation,
        groundStripBounds,
        cellSizeM,
        groundSurfaceZM,
        outputFrameIndex,
        secondsPerFrame,
    } = options;
    const crossStripCellCount = cellCountForExtent(
        groundStripBounds.min[0],
        groundStripBounds.max[0],
        cellSizeM,
    );
    const walkingCellCount = cellCountForExtent(
        groundStripBounds.min[1],
        groundStripBounds.max[1],
        cellSizeM,
    );
    const contact = new Uint8Array(
        crossStripCellCount * walkingCellCount,
    );
    const phi = unpackSdf(animation);
    const sampledWorldFromUnit = new Float32Array(
        animation.primitiveCount * MATRIX_FLOAT_COUNT,
    );

    for (
        let frameIndex = 0;
        frameIndex <= outputFrameIndex;
        frameIndex++
    ) {
        const worldFromUnit = animation.sampleWorldFromUnitAtTimeS(
            frameIndex * secondsPerFrame,
            sampledWorldFromUnit,
        );
        const expectedMatrixValues =
            animation.primitiveCount * MATRIX_FLOAT_COUNT;
        if (
            worldFromUnit.length !== expectedMatrixValues
            || worldFromUnit.some(value => !Number.isFinite(value))
        ) {
            throw new Error(
                `rigid paw pose must contain `
                + `${expectedMatrixValues} finite values`,
            );
        }

        for (
            let primitiveIndex = 0;
            primitiveIndex < animation.primitiveCount;
            primitiveIndex++
        ) {
            const matrixOffset = primitiveIndex * MATRIX_FLOAT_COUNT;
            const unitFromWorld = mat4.inverse(
                worldFromUnit.subarray(
                    matrixOffset,
                    matrixOffset + MATRIX_FLOAT_COUNT,
                ) as Mat4,
            );
            if (unitFromWorld.some(value => !Number.isFinite(value))) {
                throw new Error(
                    `rigid paw ${primitiveIndex} transform is singular`,
                );
            }

            for (let x = 0; x < crossStripCellCount; x++) {
                const worldX = groundStripBounds.min[0]
                    + (x + 0.5) * cellSizeM;
                for (let y = 0; y < walkingCellCount; y++) {
                    const contactIndex = x * walkingCellCount + y;
                    if (contact[contactIndex] !== 0) continue;

                    const worldY = groundStripBounds.min[1]
                        + (y + 0.5) * cellSizeM;
                    const unitX = unitFromWorld[0] * worldX
                        + unitFromWorld[4] * worldY
                        + unitFromWorld[8] * groundSurfaceZM
                        + unitFromWorld[12];
                    const unitY = unitFromWorld[1] * worldX
                        + unitFromWorld[5] * worldY
                        + unitFromWorld[9] * groundSurfaceZM
                        + unitFromWorld[13];
                    const unitZ = unitFromWorld[2] * worldX
                        + unitFromWorld[6] * worldY
                        + unitFromWorld[10] * groundSurfaceZM
                        + unitFromWorld[14];
                    const signedDistance = sampleSdf({
                        phi,
                        resolution: animation.sdfResolution,
                        primitiveIndex,
                        unitX,
                        unitY,
                        unitZ,
                    });
                    if (signedDistance <= 0) {
                        contact[contactIndex] = 1;
                    }
                }
            }
        }
    }

    return {
        walkingCellCount,
        crossStripCellCount,
        cellSizeM,
        contact,
        ...dilateContact({
            contact,
            crossStripCellCount,
            walkingCellCount,
        }),
    };
};
