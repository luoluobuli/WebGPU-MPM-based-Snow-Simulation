import { describe, expect, it } from "vitest";
import { ZUBEIA_GROUND_STRIP_BOUNDS } from "./ZubeiaGround";
import {
    analyzeZubeiaMatchedCachedFrameTopology,
    type ZubeiaStructuralGroundMask,
} from "./ZubeiaMatchedTopologyProof";

type TestParticle = {
    position: readonly [number, number, number],
    material?: number,
};

const CELL_SIZE_M = 1;
const CROSS_STRIP_CELL_COUNT = 6;
const WALKING_CELL_COUNT = 6;

const positionInCell = (
    x: number,
    y: number,
    z: number,
    offsetX = 0.25,
    offsetY = 0.25,
): readonly [number, number, number] => [
    ZUBEIA_GROUND_STRIP_BOUNDS.min[0] + x + offsetX,
    ZUBEIA_GROUND_STRIP_BOUNDS.min[1] + y + offsetY,
    z,
];

const frameFromParticles = (
    particles: readonly TestParticle[],
) => {
    const frame = new ArrayBuffer(particles.length * 16);
    const floats = new Float32Array(frame);
    const uints = new Uint32Array(frame);
    particles.forEach((particle, particleIndex) => {
        const offset = particleIndex * 4;
        floats[offset] = particle.position[0];
        floats[offset + 1] = particle.position[1];
        floats[offset + 2] = particle.position[2];
        uints[offset + 3] = particle.material ?? 4;
    });

    return frame;
};

const cellIndex = (x: number, y: number) =>
    x * WALKING_CELL_COUNT + y;

const maskWith = ({
    contact = [],
    dilated = contact,
    perimeter = [],
}: {
    contact?: readonly (readonly [number, number])[],
    dilated?: readonly (readonly [number, number])[],
    perimeter?: readonly (readonly [number, number])[],
} = {}): ZubeiaStructuralGroundMask => {
    const cellCount = CROSS_STRIP_CELL_COUNT * WALKING_CELL_COUNT;
    const mask = {
        walkingCellCount: WALKING_CELL_COUNT,
        crossStripCellCount: CROSS_STRIP_CELL_COUNT,
        cellSizeM: CELL_SIZE_M,
        contact: new Uint8Array(cellCount),
        dilated: new Uint8Array(cellCount),
        perimeter: new Uint8Array(cellCount),
    };
    for (const [x, y] of contact) mask.contact[cellIndex(x, y)] = 1;
    for (const [x, y] of dilated) mask.dilated[cellIndex(x, y)] = 1;
    for (const [x, y] of perimeter) mask.perimeter[cellIndex(x, y)] = 1;

    return mask;
};

const analyze = ({
    baseline,
    colliderCurrent,
    controlCurrent,
    mask = maskWith(),
}: {
    baseline: readonly TestParticle[],
    colliderCurrent: readonly TestParticle[],
    controlCurrent: readonly TestParticle[],
    mask?: ZubeiaStructuralGroundMask,
}) => {
    const baselineFrame = frameFromParticles(baseline);

    return analyzeZubeiaMatchedCachedFrameTopology({
        colliderBaselineFrame: baselineFrame,
        colliderCurrentFrame: frameFromParticles(colliderCurrent),
        controlBaselineFrame: baselineFrame.slice(0),
        controlCurrentFrame: frameFromParticles(controlCurrent),
        gridCellSizeM: CELL_SIZE_M,
        structuralGroundMask: mask,
    });
};

describe("Zubeia matched topology proof", () => {
    it("cancels global settling shared by the collider and control runs", () => {
        const baseline = [
            { position: positionInCell(1, 1, 0.5) },
            { position: positionInCell(4, 4, 0.5) },
        ];
        const settled = [
            { position: positionInCell(1, 1, -0.3) },
            { position: positionInCell(4, 4, -0.3) },
        ];

        const proof = analyze({
            baseline,
            colliderCurrent: settled,
            controlCurrent: settled,
            mask: maskWith({ contact: [[1, 1]] }),
        });

        expect(proof.farFieldVerticalResidualCenterM).toBe(0);
        expect(proof.farFieldCenteredVerticalResidualP95M).toBe(0);
        expect(proof.pawContactDownwardResidualP95M).toBe(0);
        expect(proof.maxMatchedSurfaceDepressionM).toBe(0);
        expect(proof.verticalResidualEnergyInsideDilatedFraction).toBe(0);
    });

    it("reports a local paw depression after removing far-field drift", () => {
        const baseline = [
            { position: positionInCell(1, 1, 0.5) },
            { position: positionInCell(4, 4, 0.5) },
        ];
        const colliderCurrent = [
            { position: positionInCell(1, 1, 0.2) },
            { position: positionInCell(4, 4, 0.4) },
        ];
        const controlCurrent = [
            { position: positionInCell(1, 1, 0.4) },
            { position: positionInCell(4, 4, 0.5) },
        ];

        const proof = analyze({
            baseline,
            colliderCurrent,
            controlCurrent,
            mask: maskWith({ contact: [[1, 1]] }),
        });

        expect(proof.farFieldVerticalResidualCenterM).toBeCloseTo(-0.1);
        expect(proof.farFieldCenteredVerticalResidualP95M).toBe(0);
        expect(proof.pawContactDownwardResidualP95M).toBeCloseTo(0.1);
        expect(proof.verticalResidualEnergyInsideDilatedFraction).toBe(1);
        expect(
            proof.surfaceProofMap.matchedDepressionM[cellIndex(1, 1)],
        ).toBeCloseTo(0.2);
    });
    it("does not let deep slab particles dilute a surface depression", () => {
        const baseline: TestParticle[] = [
            { position: positionInCell(1, 1, 1.5) },
            { position: positionInCell(4, 4, 1.5) },
        ];
        for (let particleIndex = 0; particleIndex < 20; particleIndex++) {
            const offsetX = 0.1 + particleIndex % 4 * 0.2;
            const offsetY = 0.1 + Math.floor(particleIndex / 4) * 0.15;
            baseline.push({
                position: positionInCell(
                    1,
                    1,
                    0.2,
                    offsetX,
                    offsetY,
                ),
            });
            baseline.push({
                position: positionInCell(
                    1,
                    1,
                    -0.2,
                    offsetX,
                    offsetY,
                ),
                material: 0,
            });
        }
        const colliderCurrent = baseline.map((particle) => ({ ...particle }));
        colliderCurrent[0] = {
            position: positionInCell(1, 1, 1.3),
        };

        const proof = analyze({
            baseline,
            colliderCurrent,
            controlCurrent: baseline,
            mask: maskWith({ contact: [[1, 1]] }),
        });

        expect(proof.pawContactDownwardResidualP95M).toBeCloseTo(0.2);
        expect(proof.farFieldCenteredVerticalResidualP95M).toBe(0);
        expect(proof.verticalResidualEnergyInsideDilatedFraction).toBe(1);
    });

    it("groups crust lifted relative to the matched control", () => {
        const baseline = [
            { position: positionInCell(1, 1, 0.5, 0.25) },
            { position: positionInCell(1, 1, 0.5, 0.75) },
            { position: positionInCell(4, 4, 0.5) },
        ];
        const colliderCurrent = [
            { position: positionInCell(1, 1, 1.25, 0.25) },
            { position: positionInCell(1, 1, 1.25, 0.75) },
            baseline[2],
        ];

        const proof = analyze({
            baseline,
            colliderCurrent,
            controlCurrent: baseline,
            mask: maskWith({ contact: [[1, 1]] }),
        });

        expect(proof.liftedCrustComponents).toHaveLength(1);
        expect(proof.liftedCrustComponents[0]).toMatchObject({
            particleCount: 2,
            particleIndices: [0, 1],
        });
        expect(
            proof.liftedCrustComponents[0].centroidLiftM,
        ).toBeCloseTo(0.75);
        expect(proof.liftedCrustComponents[0].maxLiftM).toBeCloseTo(0.75);
    });
    it("keeps a deeply settled current crust surface present", () => {
        const baseline = [
            { position: positionInCell(2, 2, 0.5) },
        ];
        const current = [
            { position: positionInCell(2, 2, -4.5) },
        ];

        const proof = analyze({
            baseline,
            colliderCurrent: current,
            controlCurrent: current,
        });
        const surface = proof.surfaceProofMap;
        const index = cellIndex(2, 2);

        expect(surface.colliderSurfacePresent[index]).toBe(1);
        expect(surface.controlSurfacePresent[index]).toBe(1);
        expect(surface.matchedSurfacePresent[index]).toBe(1);
        expect(surface.colliderTopZM[index]).toBeCloseTo(-4.5);
        expect(surface.controlTopZM[index]).toBeCloseTo(-4.5);
    });

    it("detects a matched horizontal break in a baseline neighbor edge", () => {
        const baseline = [
            { position: positionInCell(1, 1, 0.5, 0.25) },
            { position: positionInCell(1, 1, 0.5, 0.75) },
        ];
        const controlCurrent = baseline;
        const colliderCurrent = [
            baseline[0],
            { position: positionInCell(1, 1, 0.5, 1.85) },
        ];

        const proof = analyze({
            baseline,
            colliderCurrent,
            controlCurrent,
            mask: maskWith({ perimeter: [[1, 1]] }),
        });

        expect(proof.matchedBrokenEdgeCount).toBe(1);
        expect(proof.controlOpeningEdgeCount).toBe(0);
        expect(proof.crackComponents).toHaveLength(1);
        expect(proof.qualifyingCrackComponents).toHaveLength(1);
        expect(proof.crackCellMask[cellIndex(1, 1)]).toBe(1);
    });

    it("does not classify pure vertical bending as a crack", () => {
        const baseline = [
            { position: positionInCell(1, 1, 0.5, 0.25) },
            { position: positionInCell(1, 1, 0.5, 0.75) },
        ];
        const colliderCurrent = [
            baseline[0],
            { position: positionInCell(1, 1, 3.5, 0.75) },
        ];

        const proof = analyze({
            baseline,
            colliderCurrent,
            controlCurrent: baseline,
            mask: maskWith({ perimeter: [[1, 1]] }),
        });

        expect(proof.matchedBrokenEdgeCount).toBe(0);
        expect(proof.crackComponents).toEqual([]);
        expect([...proof.crackCellMask]).not.toContain(1);
    });

    it("invalidates a matched break when the control edge also opens", () => {
        const baseline = [
            { position: positionInCell(1, 1, 0.5, 0.25) },
            { position: positionInCell(1, 1, 0.5, 0.75) },
        ];
        const controlCurrent = [
            baseline[0],
            { position: positionInCell(1, 1, 0.5, 1.85) },
        ];
        const colliderCurrent = [
            baseline[0],
            { position: positionInCell(1, 1, 0.5, 2.95) },
        ];

        const proof = analyze({
            baseline,
            colliderCurrent,
            controlCurrent,
            mask: maskWith({ perimeter: [[1, 1]] }),
        });

        expect(proof.matchedBrokenEdgeCount).toBe(1);
        expect(proof.controlOpeningEdgeCount).toBe(1);
        expect(proof.controlInvalidatedBrokenEdgeCount).toBe(1);
        expect(proof.crackComponents).toEqual([]);
        expect(proof.controlOpeningCellMask[cellIndex(1, 1)]).toBe(1);
    });

    it("keeps broken-edge components disconnected unless they share a particle", () => {
        const baseline = [
            { position: positionInCell(0, 0, 0.5, 0.25) },
            { position: positionInCell(0, 0, 0.5, 0.75) },
            { position: positionInCell(4, 4, 0.5, 0.25) },
            { position: positionInCell(4, 4, 0.5, 0.75) },
        ];
        const colliderCurrent = [
            baseline[0],
            { position: positionInCell(0, 0, 0.5, 1.85) },
            baseline[2],
            { position: positionInCell(4, 4, 0.5, 1.85) },
        ];

        const proof = analyze({
            baseline,
            colliderCurrent,
            controlCurrent: baseline,
            mask: maskWith({ perimeter: [[0, 0], [4, 4]] }),
        });

        expect(proof.crackComponents).toHaveLength(2);
        expect(proof.crackComponents.map((component) => component.edgeCount))
            .toEqual([1, 1]);
        expect(proof.crackComponents.map((component) => component.particleCount))
            .toEqual([2, 2]);
    });

    it("qualifies only crack components that intersect the paw perimeter", () => {
        const baseline = [
            { position: positionInCell(3, 3, 0.5, 0.25) },
            { position: positionInCell(3, 3, 0.5, 0.75) },
        ];
        const colliderCurrent = [
            baseline[0],
            { position: positionInCell(3, 3, 0.5, 1.85) },
        ];

        const outsideProof = analyze({
            baseline,
            colliderCurrent,
            controlCurrent: baseline,
            mask: maskWith({ perimeter: [[1, 1]] }),
        });
        const perimeterProof = analyze({
            baseline,
            colliderCurrent,
            controlCurrent: baseline,
            mask: maskWith({ perimeter: [[3, 3]] }),
        });

        expect(outsideProof.crackComponents).toHaveLength(1);
        expect(outsideProof.qualifyingCrackComponents).toEqual([]);
        expect([...outsideProof.crackCellMask]).not.toContain(1);
        expect(perimeterProof.qualifyingCrackComponents).toHaveLength(1);
        expect(perimeterProof.crackCellMask[cellIndex(3, 3)]).toBe(1);
    });

    it("validates layouts, finite values, materials, and matching baselines", () => {
        const particle = { position: positionInCell(1, 1, 0.5) };
        const frame = frameFromParticles([particle]);
        const input = {
            colliderBaselineFrame: frame,
            colliderCurrentFrame: frame.slice(0),
            controlBaselineFrame: frame.slice(0),
            controlCurrentFrame: frame.slice(0),
            gridCellSizeM: CELL_SIZE_M,
            structuralGroundMask: maskWith(),
        };

        expect(() => analyzeZubeiaMatchedCachedFrameTopology({
            ...input,
            controlCurrentFrame: new ArrayBuffer(0),
        })).toThrow("equal 16-byte particle layouts");
        expect(() => analyzeZubeiaMatchedCachedFrameTopology({
            ...input,
            colliderCurrentFrame: frameFromParticles([{
                position: [Number.NaN, 0, 0],
            }]),
        })).toThrow("non-finite position");
        expect(() => analyzeZubeiaMatchedCachedFrameTopology({
            ...input,
            colliderCurrentFrame: frameFromParticles([{
                ...particle,
                material: 99,
            }]),
        })).toThrow("invalid material");
        expect(() => analyzeZubeiaMatchedCachedFrameTopology({
            ...input,
            controlBaselineFrame: frameFromParticles([{
                position: positionInCell(1, 1, 0.6),
            }]),
        })).toThrow("baselines differ");
    });
});
