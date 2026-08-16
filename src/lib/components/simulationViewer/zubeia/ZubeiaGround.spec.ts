import { describe, expect, it } from "vitest";

import {
    ZUBEIA_GROUND_BASE_APPEARANCE,
    ZUBEIA_GROUND_CELL_COUNT,
    ZUBEIA_GROUND_CELL_SIZE_M,
    ZUBEIA_GROUND_CRUST_APPEARANCE,
    ZUBEIA_GROUND_CRUST_CELL_COUNT,
    ZUBEIA_GROUND_CRUST_DEPTH_M,
    ZUBEIA_GROUND_CRUST_MATERIAL_ID,
    ZUBEIA_GROUND_DEFAULT_MATERIAL_ID,
    ZUBEIA_GROUND_DOMAIN_BOUNDS,
    ZUBEIA_GROUND_FINAL_PARTICLE_COUNT,
    ZUBEIA_GROUND_FINAL_PARTICLES_PER_CELL,
    ZUBEIA_GROUND_GRID_RESOLUTION,
    ZUBEIA_GROUND_PREVIEW_PARTICLE_COUNT,
    ZUBEIA_GROUND_PREVIEW_PARTICLES_PER_CELL,
    ZUBEIA_GROUND_STRIDE_M,
    ZUBEIA_GROUND_SURFACE_Z_M,
    ZUBEIA_GROUND_STRIP_BOUNDS,
    ZUBEIA_GROUND_STRIP_CELL_COUNTS,
    ZUBEIA_GROUND_WALK_CYCLE_COUNT,
    ZUBEIA_GROUND_WALK_DISTANCE_M,
    buildZubeiaGround,
    zubeiaGroundParticleCount,
} from "./ZubeiaGround";

const expectBoundsMatchResolution = (
    bounds: {
        min: readonly [number, number, number],
        max: readonly [number, number, number],
    },
    resolution: readonly [number, number, number],
) => {
    for (let axis = 0; axis < 3; axis++) {
        expect(bounds.max[axis] - bounds.min[axis]).toBeCloseTo(
            resolution[axis] * ZUBEIA_GROUND_CELL_SIZE_M,
        );
    }
};

describe("ZubeiaGround", () => {
    it("defines a 0.025 meter grid and a narrow strip covering three strides", () => {
        expect(ZUBEIA_GROUND_CELL_SIZE_M).toBe(0.025);
        expect(ZUBEIA_GROUND_SURFACE_Z_M).toBeCloseTo(0.07);
        expect(ZUBEIA_GROUND_GRID_RESOLUTION).toEqual([80, 192, 41]);
        expect(ZUBEIA_GROUND_STRIP_CELL_COUNTS).toEqual([68, 176, 11]);
        expect(ZUBEIA_GROUND_DOMAIN_BOUNDS.min[2]).toBeCloseTo(-0.255);
        expect(ZUBEIA_GROUND_DOMAIN_BOUNDS.max[2]).toBeCloseTo(0.77);
        expect(ZUBEIA_GROUND_STRIP_BOUNDS.max[2]).toBeCloseTo(
            ZUBEIA_GROUND_SURFACE_Z_M,
        );
        expect(
            ZUBEIA_GROUND_DOMAIN_BOUNDS.min[2]
            + 2 * ZUBEIA_GROUND_CELL_SIZE_M,
        ).toBeCloseTo(ZUBEIA_GROUND_STRIP_BOUNDS.min[2]);
        expectBoundsMatchResolution(
            ZUBEIA_GROUND_DOMAIN_BOUNDS,
            ZUBEIA_GROUND_GRID_RESOLUTION,
        );
        expectBoundsMatchResolution(
            ZUBEIA_GROUND_STRIP_BOUNDS,
            ZUBEIA_GROUND_STRIP_CELL_COUNTS,
        );

        expect(ZUBEIA_GROUND_WALK_CYCLE_COUNT).toBe(3);
        expect(ZUBEIA_GROUND_WALK_DISTANCE_M).toBeCloseTo(
            ZUBEIA_GROUND_WALK_CYCLE_COUNT * ZUBEIA_GROUND_STRIDE_M,
        );
        expect(
            ZUBEIA_GROUND_STRIP_BOUNDS.max[1]
            - ZUBEIA_GROUND_STRIP_BOUNDS.min[1],
        ).toBeGreaterThan(ZUBEIA_GROUND_WALK_DISTANCE_M);
        expect(ZUBEIA_GROUND_STRIP_BOUNDS.min[1]).toBeLessThanOrEqual(-1.2);
        expect(ZUBEIA_GROUND_STRIP_BOUNDS.max[1]).toBeGreaterThanOrEqual(3.0);
    });

    it("exports profile counts derived from one and four particles per cell", () => {
        expect(ZUBEIA_GROUND_CELL_COUNT).toBe(68 * 176 * 11);
        expect(ZUBEIA_GROUND_PREVIEW_PARTICLES_PER_CELL).toBe(1);
        expect(ZUBEIA_GROUND_FINAL_PARTICLES_PER_CELL).toBe(4);
        expect(ZUBEIA_GROUND_PREVIEW_PARTICLE_COUNT).toBe(131_648);
        expect(ZUBEIA_GROUND_FINAL_PARTICLE_COUNT).toBe(526_592);
        expect(zubeiaGroundParticleCount("preview")).toBe(
            ZUBEIA_GROUND_PREVIEW_PARTICLE_COUNT,
        );
        expect(zubeiaGroundParticleCount("final")).toBe(
            ZUBEIA_GROUND_FINAL_PARTICLE_COUNT,
        );
    });

    it("places one deterministic center sample in every preview cell", () => {
        const first = buildZubeiaGround("preview");
        const second = buildZubeiaGround("preview");

        expect(first.points).toEqual(second.points);
        expect(first.particleAppearances).toEqual(second.particleAppearances);
        expect(first.points.length).toBe(
            ZUBEIA_GROUND_PREVIEW_PARTICLE_COUNT * 4,
        );
        expect(first.points[0]).toBeCloseTo(
            ZUBEIA_GROUND_STRIP_BOUNDS.min[0] + ZUBEIA_GROUND_CELL_SIZE_M * 0.5,
        );
        expect(first.points[1]).toBeCloseTo(
            ZUBEIA_GROUND_STRIP_BOUNDS.min[1] + ZUBEIA_GROUND_CELL_SIZE_M * 0.5,
        );
        expect(first.points[2]).toBeCloseTo(
            ZUBEIA_GROUND_STRIP_BOUNDS.min[2] + ZUBEIA_GROUND_CELL_SIZE_M * 0.5,
        );
    });

    it("places a regular tetrahedron of four samples in every final cell", () => {
        const ground = buildZubeiaGround("final");
        const firstCellOffsets = [
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ];

        expect(ground.points.length).toBe(
            ZUBEIA_GROUND_FINAL_PARTICLE_COUNT * 4,
        );

        for (
            let particleIndex = 0;
            particleIndex < firstCellOffsets.length;
            particleIndex++
        ) {
            const expectedOffset = firstCellOffsets[particleIndex];
            const pointOffset = particleIndex * 4;

            for (let axis = 0; axis < 3; axis++) {
                expect(ground.points[pointOffset + axis]).toBeCloseTo(
                    ZUBEIA_GROUND_STRIP_BOUNDS.min[axis]
                    + expectedOffset[axis] * ZUBEIA_GROUND_CELL_SIZE_M,
                );
            }
        }
    });

    it("encodes only the top cell layer as physical crust material", () => {
        const ground = buildZubeiaGround("final");
        let baseCount = 0;
        let crustCount = 0;
        let baseMaterialsMatch = true;
        let baseAppearancesMatch = true;
        let crustAppearancesMatch = true;

        for (
            let particleIndex = 0;
            particleIndex < ground.particleAppearances.length;
            particleIndex++
        ) {
            const material = ground.points[particleIndex * 4 + 3];
            const appearance = ground.particleAppearances[particleIndex];

            if (material === ZUBEIA_GROUND_CRUST_MATERIAL_ID) {
                crustCount++;
                crustAppearancesMatch =
                    crustAppearancesMatch
                    && appearance === ZUBEIA_GROUND_CRUST_APPEARANCE;
            } else {
                baseCount++;
                baseMaterialsMatch =
                    baseMaterialsMatch
                    && material === ZUBEIA_GROUND_DEFAULT_MATERIAL_ID;
                baseAppearancesMatch =
                    baseAppearancesMatch
                    && appearance === ZUBEIA_GROUND_BASE_APPEARANCE;
            }
        }

        expect(baseMaterialsMatch).toBe(true);
        expect(baseAppearancesMatch).toBe(true);
        expect(crustAppearancesMatch).toBe(true);

        const crustCellCount =
            ZUBEIA_GROUND_STRIP_CELL_COUNTS[0]
            * ZUBEIA_GROUND_STRIP_CELL_COUNTS[1]
            * ZUBEIA_GROUND_CRUST_CELL_COUNT;
        expect(ZUBEIA_GROUND_CRUST_DEPTH_M).toBeCloseTo(0.05);
        expect(crustCount).toBe(
            crustCellCount * ZUBEIA_GROUND_FINAL_PARTICLES_PER_CELL,
        );
        expect(baseCount + crustCount).toBe(
            ZUBEIA_GROUND_FINAL_PARTICLE_COUNT,
        );
    });

    it("uses distinct packed colors for the snow base and crunchy crust", () => {
        expect(ZUBEIA_GROUND_BASE_APPEARANCE).not.toBe(
            ZUBEIA_GROUND_CRUST_APPEARANCE,
        );
        expect(ZUBEIA_GROUND_BASE_APPEARANCE >>> 24).toBeGreaterThan(0);
        expect(ZUBEIA_GROUND_CRUST_APPEARANCE >>> 24).toBeGreaterThan(0);
    });
});
