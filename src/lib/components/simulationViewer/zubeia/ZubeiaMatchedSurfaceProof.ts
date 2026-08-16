import type { ZubeiaPawContactMask } from "./ZubeiaPawContactMask";
import { ZUBEIA_GROUND_CRUST_MATERIAL_ID } from "./ZubeiaGround";
import {
    type CachedFrameView,
    componentAt,
    maskIndexForPosition,
    materialAt,
} from "./ZubeiaMatchedTopologyFrame";

const buildSurface = (
    frame: CachedFrameView,
    particleCount: number,
    mask: ZubeiaPawContactMask,
) => {
    const cellCount = mask.walkingCellCount * mask.crossStripCellCount;
    const present = new Uint8Array(cellCount);
    const topZM = new Float32Array(cellCount);
    topZM.fill(Number.NEGATIVE_INFINITY);

    for (let particleIndex = 0; particleIndex < particleCount; particleIndex++) {
        if (materialAt(frame, particleIndex) !== ZUBEIA_GROUND_CRUST_MATERIAL_ID) {
            continue;
        }

        const cellIndex = maskIndexForPosition(
            componentAt(frame, particleIndex, 0),
            componentAt(frame, particleIndex, 1),
            mask,
        );
        if (cellIndex < 0) continue;

        present[cellIndex] = 1;
        topZM[cellIndex] = Math.max(
            topZM[cellIndex],
            componentAt(frame, particleIndex, 2),
        );
    }

    for (let cellIndex = 0; cellIndex < cellCount; cellIndex++) {
        if (present[cellIndex] === 0) topZM[cellIndex] = 0;
    }

    return {
        present,
        topZM,
    };
};

export const buildMatchedSurfaceProof = ({
    colliderCurrent,
    controlCurrent,
    particleCount,
    mask,
}: {
    colliderCurrent: CachedFrameView,
    controlCurrent: CachedFrameView,
    particleCount: number,
    mask: ZubeiaPawContactMask,
}) => {
    const collider = buildSurface(colliderCurrent, particleCount, mask);
    const control = buildSurface(controlCurrent, particleCount, mask);
    const cellCount = mask.walkingCellCount * mask.crossStripCellCount;
    const matchedSurfacePresent = new Uint8Array(cellCount);
    const matchedDepressionM = new Float32Array(cellCount);
    let maxMatchedSurfaceDepressionM = 0;

    for (let cellIndex = 0; cellIndex < cellCount; cellIndex++) {
        if (
            collider.present[cellIndex] === 0
            || control.present[cellIndex] === 0
        ) {
            continue;
        }

        matchedSurfacePresent[cellIndex] = 1;
        const depressionM = Math.max(
            0,
            control.topZM[cellIndex] - collider.topZM[cellIndex],
        );
        matchedDepressionM[cellIndex] = depressionM;
        maxMatchedSurfaceDepressionM = Math.max(
            maxMatchedSurfaceDepressionM,
            depressionM,
        );
    }

    return {
        maxMatchedSurfaceDepressionM,
        surfaceProofMap: {
            walkingCellCount: mask.walkingCellCount,
            crossStripCellCount: mask.crossStripCellCount,
            cellSizeM: mask.cellSizeM,
            colliderSurfacePresent: collider.present,
            controlSurfacePresent: control.present,
            matchedSurfacePresent,
            colliderTopZM: collider.topZM,
            controlTopZM: control.topZM,
            matchedDepressionM,
        },
    };
};
