import type { ZubeiaPawContactMask } from "./ZubeiaPawContactMask";
import {
    ZUBEIA_GROUND_CRUST_MATERIAL_ID,
    ZUBEIA_GROUND_STRIP_BOUNDS,
} from "./ZubeiaGround";

export type CachedFrameView = {
    floats: Float32Array,
    uints: Uint32Array,
};

export const MATCHED_CACHED_PARTICLE_BYTE_LENGTH = 16;
export const POSITION_QUANTIZATION_EPSILON = 1e-4;

const POSITION_COMPONENTS_PER_PARTICLE = 4;

const frameView = (frame: ArrayBuffer): CachedFrameView => ({
    floats: new Float32Array(frame),
    uints: new Uint32Array(frame),
});

export const componentAt = (
    frame: CachedFrameView,
    particleIndex: number,
    axis: number,
) => frame.floats[
    particleIndex * POSITION_COMPONENTS_PER_PARTICLE + axis
];

export const materialAt = (
    frame: CachedFrameView,
    particleIndex: number,
) => frame.uints[
    particleIndex * POSITION_COMPONENTS_PER_PARTICLE + 3
];

export const positionAt = (
    frame: CachedFrameView,
    particleIndex: number,
): readonly [number, number, number] => [
    componentAt(frame, particleIndex, 0),
    componentAt(frame, particleIndex, 1),
    componentAt(frame, particleIndex, 2),
];

export const quantize = (
    value: number,
    origin: number,
    cellSizeM: number,
) => Math.floor(
    (value - origin) / cellSizeM + POSITION_QUANTIZATION_EPSILON,
);

export const maskIndexForCell = (
    x: number,
    y: number,
    mask: ZubeiaPawContactMask,
) => {
    if (
        x < 0
        || x >= mask.crossStripCellCount
        || y < 0
        || y >= mask.walkingCellCount
    ) {
        return -1;
    }

    return x * mask.walkingCellCount + y;
};

export const maskIndexForPosition = (
    xM: number,
    yM: number,
    mask: ZubeiaPawContactMask,
) => maskIndexForCell(
    quantize(
        xM,
        ZUBEIA_GROUND_STRIP_BOUNDS.min[0],
        mask.cellSizeM,
    ),
    quantize(
        yM,
        ZUBEIA_GROUND_STRIP_BOUNDS.min[1],
        mask.cellSizeM,
    ),
    mask,
);

export const sortedMedian = (
    values: Float64Array,
    valueCount: number,
) => {
    if (valueCount === 0) return 0;

    const sorted = values.subarray(0, valueCount);
    sorted.sort();
    const upperIndex = Math.floor(valueCount / 2);
    if (valueCount % 2 === 1) return sorted[upperIndex];

    return (sorted[upperIndex - 1] + sorted[upperIndex]) / 2;
};

export const sortedP95 = (
    values: Float64Array,
    valueCount: number,
) => {
    if (valueCount === 0) return 0;

    const sorted = values.subarray(0, valueCount);
    sorted.sort();

    return sorted[Math.ceil(valueCount * 0.95) - 1];
};

export const validateMatchedMask = (
    mask: ZubeiaPawContactMask,
    gridCellSizeM: number,
) => {
    if (
        !Number.isInteger(mask.walkingCellCount)
        || !Number.isInteger(mask.crossStripCellCount)
        || mask.walkingCellCount <= 0
        || mask.crossStripCellCount <= 0
    ) {
        throw new Error(
            "Zubeia structural mask dimensions must be positive integers",
        );
    }
    if (!Number.isFinite(mask.cellSizeM) || mask.cellSizeM <= 0) {
        throw new Error("Zubeia structural mask cell size must be positive");
    }
    if (
        Math.abs(mask.cellSizeM - gridCellSizeM)
        > gridCellSizeM * 1e-6
    ) {
        throw new Error(
            "Zubeia structural mask and simulation cell sizes must match",
        );
    }

    const expectedLength =
        mask.walkingCellCount * mask.crossStripCellCount;
    for (const [label, values] of [
        ["contact", mask.contact],
        ["dilated", mask.dilated],
        ["perimeter", mask.perimeter],
    ] as const) {
        if (!(values instanceof Uint8Array) || values.length !== expectedLength) {
            throw new Error(
                `Zubeia structural ${label} mask must contain ${expectedLength} cells`,
            );
        }
    }
};

export const validateMatchedFrames = (
    frames: readonly [ArrayBuffer, ArrayBuffer, ArrayBuffer, ArrayBuffer],
) => {
    const byteLength = frames[0].byteLength;
    if (
        byteLength % MATCHED_CACHED_PARTICLE_BYTE_LENGTH !== 0
        || frames.some((frame) => frame.byteLength !== byteLength)
    ) {
        throw new Error(
            "Zubeia matched topology frames must have equal 16-byte particle layouts",
        );
    }

    const views: readonly [
        CachedFrameView,
        CachedFrameView,
        CachedFrameView,
        CachedFrameView,
    ] = [
        frameView(frames[0]),
        frameView(frames[1]),
        frameView(frames[2]),
        frameView(frames[3]),
    ];
    const particleCount =
        byteLength / MATCHED_CACHED_PARTICLE_BYTE_LENGTH;
    const labels = [
        "collider baseline",
        "collider current",
        "control baseline",
        "control current",
    ];

    for (let particleIndex = 0; particleIndex < particleCount; particleIndex++) {
        for (let frameIndex = 0; frameIndex < views.length; frameIndex++) {
            const view = views[frameIndex];
            for (let axis = 0; axis < 3; axis++) {
                if (Number.isFinite(componentAt(view, particleIndex, axis))) {
                    continue;
                }

                throw new Error(
                    `${labels[frameIndex]} cached particle ${particleIndex} has a non-finite position`,
                );
            }

            if (
                materialAt(view, particleIndex)
                <= ZUBEIA_GROUND_CRUST_MATERIAL_ID
            ) {
                continue;
            }

            throw new Error(
                `${labels[frameIndex]} cached particle ${particleIndex} has an invalid material`,
            );
        }

        for (let axis = 0; axis < 3; axis++) {
            if (
                componentAt(views[0], particleIndex, axis)
                === componentAt(views[2], particleIndex, axis)
            ) {
                continue;
            }

            throw new Error(
                `Zubeia collider and control baselines differ at particle ${particleIndex}`,
            );
        }

        const baselineMaterial = materialAt(views[0], particleIndex);
        if (
            materialAt(views[2], particleIndex) !== baselineMaterial
            || materialAt(views[1], particleIndex) !== baselineMaterial
            || materialAt(views[3], particleIndex) !== baselineMaterial
        ) {
            throw new Error(
                `Zubeia cached particle ${particleIndex} changed material`,
            );
        }
    }

    return {
        views,
        particleCount,
    };
};
export const buildBaselineTopCrustParticleMask = ({
    baseline,
    particleCount,
    mask,
    gridCellSizeM,
}: {
    baseline: CachedFrameView,
    particleCount: number,
    mask: ZubeiaPawContactMask,
    gridCellSizeM: number,
}) => {
    const cellCount = mask.walkingCellCount * mask.crossStripCellCount;
    const topCellZ = new Int32Array(cellCount);
    topCellZ.fill(-0x80000000);

    for (let particleIndex = 0; particleIndex < particleCount; particleIndex++) {
        if (
            materialAt(baseline, particleIndex)
            !== ZUBEIA_GROUND_CRUST_MATERIAL_ID
        ) {
            continue;
        }

        const cellIndex = maskIndexForPosition(
            componentAt(baseline, particleIndex, 0),
            componentAt(baseline, particleIndex, 1),
            mask,
        );
        if (cellIndex < 0) continue;

        const z = quantize(
            componentAt(baseline, particleIndex, 2),
            ZUBEIA_GROUND_STRIP_BOUNDS.min[2],
            gridCellSizeM,
        );
        topCellZ[cellIndex] = Math.max(topCellZ[cellIndex], z);
    }

    const topCrust = new Uint8Array(particleCount);
    for (let particleIndex = 0; particleIndex < particleCount; particleIndex++) {
        if (
            materialAt(baseline, particleIndex)
            !== ZUBEIA_GROUND_CRUST_MATERIAL_ID
        ) {
            continue;
        }

        const cellIndex = maskIndexForPosition(
            componentAt(baseline, particleIndex, 0),
            componentAt(baseline, particleIndex, 1),
            mask,
        );
        if (cellIndex < 0) continue;

        const z = quantize(
            componentAt(baseline, particleIndex, 2),
            ZUBEIA_GROUND_STRIP_BOUNDS.min[2],
            gridCellSizeM,
        );
        if (z === topCellZ[cellIndex]) topCrust[particleIndex] = 1;
    }

    return topCrust;
};
