import {
    ZUBEIA_WALK_GROUND_SURFACE_Z_M,
    ZUBEIA_WALK_STRIDE_M,
} from "./ZubeiaWalkCalibration";

type Vec3 = readonly [number, number, number];

export type ZubeiaGroundBounds = {
    min: Vec3,
    max: Vec3,
};

export type ZubeiaGroundData = {
    points: Float32Array,
};

export const ZUBEIA_GROUND_CELL_SIZE_M = 0.025;
export const ZUBEIA_GROUND_SURFACE_Z_M =
    ZUBEIA_WALK_GROUND_SURFACE_Z_M;
export const ZUBEIA_GROUND_WALK_CYCLE_COUNT = 3;
export const ZUBEIA_GROUND_STRIDE_M = ZUBEIA_WALK_STRIDE_M;
export const ZUBEIA_GROUND_WALK_DISTANCE_M =
    ZUBEIA_GROUND_WALK_CYCLE_COUNT * ZUBEIA_GROUND_STRIDE_M;

export const ZUBEIA_GROUND_DOMAIN_BOUNDS: ZubeiaGroundBounds = {
    min: [-1, -1.5, -0.255],
    max: [1, 3.3, 0.77],
};

export const ZUBEIA_GROUND_STRIP_BOUNDS: ZubeiaGroundBounds = {
    min: [-0.85, -1.3, -0.205],
    max: [0.85, 3.1, ZUBEIA_GROUND_SURFACE_Z_M],
};

const cellCountsForBounds = ({
    min,
    max,
}: ZubeiaGroundBounds): [number, number, number] => [
    Math.round((max[0] - min[0]) / ZUBEIA_GROUND_CELL_SIZE_M),
    Math.round((max[1] - min[1]) / ZUBEIA_GROUND_CELL_SIZE_M),
    Math.round((max[2] - min[2]) / ZUBEIA_GROUND_CELL_SIZE_M),
];

export const ZUBEIA_GROUND_GRID_RESOLUTION = cellCountsForBounds(
    ZUBEIA_GROUND_DOMAIN_BOUNDS,
);
export const ZUBEIA_GROUND_STRIP_CELL_COUNTS = cellCountsForBounds(
    ZUBEIA_GROUND_STRIP_BOUNDS,
);

export const ZUBEIA_GROUND_PARTICLE_COUNT =
    ZUBEIA_GROUND_STRIP_CELL_COUNTS[0]
    * ZUBEIA_GROUND_STRIP_CELL_COUNTS[1]
    * ZUBEIA_GROUND_STRIP_CELL_COUNTS[2];

export const buildZubeiaGround = (): ZubeiaGroundData => {
    const points = new Float32Array(ZUBEIA_GROUND_PARTICLE_COUNT * 4);
    const [cellCountX, cellCountY, cellCountZ] =
        ZUBEIA_GROUND_STRIP_CELL_COUNTS;
    let particleIndex = 0;

    for (let z = 0; z < cellCountZ; z++) {
        for (let y = 0; y < cellCountY; y++) {
            for (let x = 0; x < cellCountX; x++) {
                const pointOffset = particleIndex * 4;
                points[pointOffset] =
                    ZUBEIA_GROUND_STRIP_BOUNDS.min[0]
                    + (x + 0.5) * ZUBEIA_GROUND_CELL_SIZE_M;
                points[pointOffset + 1] =
                    ZUBEIA_GROUND_STRIP_BOUNDS.min[1]
                    + (y + 0.5) * ZUBEIA_GROUND_CELL_SIZE_M;
                points[pointOffset + 2] =
                    ZUBEIA_GROUND_STRIP_BOUNDS.min[2]
                    + (z + 0.5) * ZUBEIA_GROUND_CELL_SIZE_M;
                points[pointOffset + 3] = 1;
                particleIndex++;
            }
        }
    }

    return {
        points,
    };
};
