import {
    ParticleAppearanceMaterial,
    packParticleAppearance,
} from "$lib/gpu/particleAppearance/GpuParticleAppearanceBufferManager";
import {
    ZUBEIA_WALK_GROUND_SURFACE_Z_M,
    ZUBEIA_WALK_STRIDE_M,
} from "./ZubeiaWalkCalibration";

type Vec3 = readonly [number, number, number];

export type ZubeiaGroundBounds = {
    min: Vec3,
    max: Vec3,
};

export type ZubeiaGroundProfile = "preview" | "final";

export type ZubeiaGroundData = {
    points: Float32Array,
    particleAppearances: Uint32Array,
};

export const ZUBEIA_GROUND_CELL_SIZE_M = 0.025;
export const ZUBEIA_GROUND_CRUST_CELL_COUNT = 2;
export const ZUBEIA_GROUND_CRUST_DEPTH_M =
    ZUBEIA_GROUND_CRUST_CELL_COUNT * ZUBEIA_GROUND_CELL_SIZE_M;
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

export const ZUBEIA_GROUND_DEFAULT_MATERIAL_ID = 0;
export const ZUBEIA_GROUND_CRUST_MATERIAL_ID = 4;
export const ZUBEIA_GROUND_PREVIEW_PARTICLES_PER_CELL = 1;
export const ZUBEIA_GROUND_FINAL_PARTICLES_PER_CELL = 4;

export const ZUBEIA_GROUND_CELL_COUNT =
    ZUBEIA_GROUND_STRIP_CELL_COUNTS[0]
    * ZUBEIA_GROUND_STRIP_CELL_COUNTS[1]
    * ZUBEIA_GROUND_STRIP_CELL_COUNTS[2];
export const ZUBEIA_GROUND_PREVIEW_PARTICLE_COUNT =
    ZUBEIA_GROUND_CELL_COUNT * ZUBEIA_GROUND_PREVIEW_PARTICLES_PER_CELL;
export const ZUBEIA_GROUND_FINAL_PARTICLE_COUNT =
    ZUBEIA_GROUND_CELL_COUNT * ZUBEIA_GROUND_FINAL_PARTICLES_PER_CELL;

export const ZUBEIA_GROUND_BASE_APPEARANCE = packParticleAppearance({
    color: [0.29, 0.18, 0.11],
    material: ParticleAppearanceMaterial.Soil,
}) >>> 0;
export const ZUBEIA_GROUND_CRUST_APPEARANCE = packParticleAppearance({
    color: [0.3, 0.34, 0.37],
    material: ParticleAppearanceMaterial.Soil,
}) >>> 0;

const PREVIEW_SAMPLE_OFFSETS: readonly Vec3[] = [
    [0.5, 0.5, 0.5],
];

// These four points form a regular tetrahedron centered in the cell. This
// deterministic layout supplies four particles per cell without coincident
// samples or a random seed.
const FINAL_SAMPLE_OFFSETS: readonly Vec3[] = [
    [0.25, 0.25, 0.25],
    [0.75, 0.75, 0.25],
    [0.75, 0.25, 0.75],
    [0.25, 0.75, 0.75],
];

const sampleOffsetsForProfile = (profile: ZubeiaGroundProfile) =>
    profile === "final"
        ? FINAL_SAMPLE_OFFSETS
        : PREVIEW_SAMPLE_OFFSETS;

export const zubeiaGroundParticleCount = (profile: ZubeiaGroundProfile) =>
    profile === "final"
        ? ZUBEIA_GROUND_FINAL_PARTICLE_COUNT
        : ZUBEIA_GROUND_PREVIEW_PARTICLE_COUNT;

export const buildZubeiaGround = (
    profile: ZubeiaGroundProfile,
): ZubeiaGroundData => {
    const sampleOffsets = sampleOffsetsForProfile(profile);
    const particleCount = zubeiaGroundParticleCount(profile);
    const points = new Float32Array(particleCount * 4);
    const particleAppearances = new Uint32Array(particleCount);
    const [cellCountX, cellCountY, cellCountZ] =
        ZUBEIA_GROUND_STRIP_CELL_COUNTS;
    const crustStartZ = cellCountZ - ZUBEIA_GROUND_CRUST_CELL_COUNT;
    let particleIndex = 0;

    for (let z = 0; z < cellCountZ; z++) {
        const isCrust = z >= crustStartZ;
        const material = isCrust
            ? ZUBEIA_GROUND_CRUST_MATERIAL_ID
            : ZUBEIA_GROUND_DEFAULT_MATERIAL_ID;
        const appearance = isCrust
            ? ZUBEIA_GROUND_CRUST_APPEARANCE
            : ZUBEIA_GROUND_BASE_APPEARANCE;

        for (let y = 0; y < cellCountY; y++) {
            for (let x = 0; x < cellCountX; x++) {
                for (const offset of sampleOffsets) {
                    const pointOffset = particleIndex * 4;
                    points[pointOffset] =
                        ZUBEIA_GROUND_STRIP_BOUNDS.min[0]
                        + (x + offset[0]) * ZUBEIA_GROUND_CELL_SIZE_M;
                    points[pointOffset + 1] =
                        ZUBEIA_GROUND_STRIP_BOUNDS.min[1]
                        + (y + offset[1]) * ZUBEIA_GROUND_CELL_SIZE_M;
                    points[pointOffset + 2] =
                        ZUBEIA_GROUND_STRIP_BOUNDS.min[2]
                        + (z + offset[2]) * ZUBEIA_GROUND_CELL_SIZE_M;
                    points[pointOffset + 3] = material;
                    particleAppearances[particleIndex] = appearance;
                    particleIndex++;
                }
            }
        }
    }

    return {
        points,
        particleAppearances,
    };
};
