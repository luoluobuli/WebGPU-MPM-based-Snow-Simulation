import { describe, expect, it } from "vitest";
import type { ZubeiaMatchedTopologyProof } from "./ZubeiaMatchedTopologyProof";
import type { ZubeiaPawContactMask } from "./ZubeiaPawContactMask";
import {
    ZUBEIA_MATCHED_MAP_COLORS,
    ZUBEIA_MATCHED_MAP_DEPTH_MAX_M,
    ZUBEIA_MATCHED_MAP_PIXELS_PER_CELL,
    renderZubeiaMatchedSurfacePixels,
} from "./ZubeiaSurfaceProofMap.svelte";
import mapSource from "./ZubeiaSurfaceProofMap.svelte?raw";

const maskWith = ({
    cellCount,
    contact = [],
    perimeter = [],
}: {
    cellCount: number,
    contact?: readonly number[],
    perimeter?: readonly number[],
}): ZubeiaPawContactMask => {
    const mask = {
        walkingCellCount: cellCount,
        crossStripCellCount: 1,
        cellSizeM: 0.025,
        contact: new Uint8Array(cellCount),
        dilated: new Uint8Array(cellCount),
        perimeter: new Uint8Array(cellCount),
    };
    for (const index of contact) {
        mask.contact[index] = 1;
        mask.dilated[index] = 1;
    }
    for (const index of perimeter) {
        mask.perimeter[index] = 1;
        mask.dilated[index] = 1;
    }

    return mask;
};

const proofWith = ({
    colliderPresent,
    controlPresent,
    colliderTopZM,
    controlTopZM,
    crackCells = [],
    controlOpeningCells = [],
}: {
    colliderPresent: readonly number[],
    controlPresent: readonly number[],
    colliderTopZM: readonly number[],
    controlTopZM: readonly number[],
    crackCells?: readonly number[],
    controlOpeningCells?: readonly number[],
}): ZubeiaMatchedTopologyProof => {
    const cellCount = colliderPresent.length;
    const matchedSurfacePresent = new Uint8Array(cellCount);
    const matchedDepressionM = new Float32Array(cellCount);
    const crackCellMask = new Uint8Array(cellCount);
    const controlOpeningCellMask = new Uint8Array(cellCount);
    for (let index = 0; index < cellCount; index++) {
        matchedSurfacePresent[index] =
            colliderPresent[index] && controlPresent[index] ? 1 : 0;
        matchedDepressionM[index] = Math.max(
            0,
            controlTopZM[index] - colliderTopZM[index],
        );
    }
    for (const index of crackCells) crackCellMask[index] = 1;
    for (const index of controlOpeningCells) {
        controlOpeningCellMask[index] = 1;
    }

    return {
        particleCount: 0,
        crustParticleCount: 0,
        farFieldVerticalResidualCenterM: 0,
        farFieldCenteredVerticalResidualP95M: 0,
        pawContactDownwardResidualP95M: 0,
        verticalResidualEnergyInsideDilatedFraction: 0,
        maxMatchedSurfaceDepressionM: Math.max(...matchedDepressionM),
        surfaceProofMap: {
            walkingCellCount: cellCount,
            crossStripCellCount: 1,
            cellSizeM: 0.025,
            colliderSurfacePresent: Uint8Array.from(colliderPresent),
            controlSurfacePresent: Uint8Array.from(controlPresent),
            matchedSurfacePresent,
            colliderTopZM: Float32Array.from(colliderTopZM),
            controlTopZM: Float32Array.from(controlTopZM),
            matchedDepressionM,
        },
        liftedCrustComponents: [],
        matchedBrokenEdgeCount: 0,
        controlOpeningEdgeCount: 0,
        controlInvalidatedBrokenEdgeCount: 0,
        crackComponents: [],
        qualifyingCrackComponents: [],
        crackCellMask,
        controlOpeningCellMask,
    };
};

const pixelAt = (
    rendered: ReturnType<typeof renderZubeiaMatchedSurfacePixels>,
    walkingIndex: number,
    localX = 2,
    localY = 2,
) => {
    const x = walkingIndex * ZUBEIA_MATCHED_MAP_PIXELS_PER_CELL + localX;
    const offset = (localY * rendered.width + x) * 4;

    return [...rendered.pixels.slice(offset, offset + 3)];
};

describe("Zubeia matched surface proof map", () => {
    it("colors control-top minus collider-top depth and clamps at 75 mm", () => {
        const proof = proofWith({
            colliderPresent: [1, 1, 1, 1],
            controlPresent: [1, 1, 1, 1],
            colliderTopZM: [0, 0, 0, 0],
            controlTopZM: [0, 0.0375, 0.075, 0.2],
        });
        const rendered = renderZubeiaMatchedSurfacePixels({
            proof,
            pawContactMask: maskWith({ cellCount: 4 }),
        });

        expect(pixelAt(rendered, 0)).toEqual([205, 222, 235]);
        expect(pixelAt(rendered, 2)).toEqual([235, 63, 79]);
        expect(pixelAt(rendered, 3)).toEqual(pixelAt(rendered, 2));
        expect(ZUBEIA_MATCHED_MAP_DEPTH_MAX_M).toBe(0.075);
    });

    it("uses black only for a control surface missing in the collider run", () => {
        const proof = proofWith({
            colliderPresent: [0, 0, 1, 1],
            controlPresent: [1, 0, 0, 1],
            colliderTopZM: [0, 0, 0, 0],
            controlTopZM: [0, 0, 0, 0],
        });
        const rendered = renderZubeiaMatchedSurfacePixels({
            proof,
            pawContactMask: maskWith({ cellCount: 4 }),
        });

        expect(pixelAt(rendered, 0)).toEqual(
            [...ZUBEIA_MATCHED_MAP_COLORS.missingColliderSurface],
        );
        expect(pixelAt(rendered, 1)).toEqual(
            [...ZUBEIA_MATCHED_MAP_COLORS.noMatchedSurface],
        );
        expect(pixelAt(rendered, 2)).toEqual(
            [...ZUBEIA_MATCHED_MAP_COLORS.noMatchedSurface],
        );
        expect([0, 1, 2, 3].filter(
            (index) => pixelAt(rendered, index).every(value => value === 0),
        )).toEqual([0]);
    });

    it("renders contact, perimeter, cracks, and control opening distinctly", () => {
        const proof = proofWith({
            colliderPresent: [1, 1, 1, 1],
            controlPresent: [1, 1, 1, 1],
            colliderTopZM: [0, 0, 0, 0],
            controlTopZM: [0, 0, 0, 0],
            crackCells: [2],
            controlOpeningCells: [3],
        });
        const rendered = renderZubeiaMatchedSurfacePixels({
            proof,
            pawContactMask: maskWith({
                cellCount: 4,
                contact: [0],
                perimeter: [1],
            }),
        });

        expect(pixelAt(rendered, 0, 0, 0)).toEqual(
            [...ZUBEIA_MATCHED_MAP_COLORS.pawContact],
        );
        expect(pixelAt(rendered, 1, 0, 0)).toEqual(
            [...ZUBEIA_MATCHED_MAP_COLORS.pawPerimeter],
        );
        expect(pixelAt(rendered, 2, 0, 0)).toEqual(
            [...ZUBEIA_MATCHED_MAP_COLORS.qualifyingCrack],
        );
        expect(pixelAt(rendered, 3, 0, 0)).toEqual(
            [...ZUBEIA_MATCHED_MAP_COLORS.controlOpening],
        );
    });

    it("states the matched meaning and exposes only matched proof props", () => {
        expect(mapSource).toContain("proof: ZubeiaMatchedTopologyProof");
        expect(mapSource).toContain("pawContactMask: ZubeiaPawContactMask");
        expect(mapSource).toContain("control top Z minus collider top");
        expect(mapSource).toContain("All evidence is collider minus");
        expect(mapSource).toContain("qualifying crack");
        expect(mapSource).toContain("control-run opening");
        expect(mapSource).not.toContain("currentSurfaceDensityRatio");
        expect(mapSource).not.toContain("Gaps are not classified as cracks");
    });

    it("rejects a paw mask with different map dimensions", () => {
        const proof = proofWith({
            colliderPresent: [1, 1],
            controlPresent: [1, 1],
            colliderTopZM: [0, 0],
            controlTopZM: [0, 0],
        });

        expect(() => renderZubeiaMatchedSurfacePixels({
            proof,
            pawContactMask: maskWith({ cellCount: 3 }),
        })).toThrow("dimensions must match");
    });
});
