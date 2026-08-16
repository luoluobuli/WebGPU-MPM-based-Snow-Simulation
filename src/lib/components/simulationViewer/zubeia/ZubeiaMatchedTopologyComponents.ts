import type { ZubeiaPawContactMask } from "./ZubeiaPawContactMask";
import { ZUBEIA_GROUND_CRUST_MATERIAL_ID } from "./ZubeiaGround";
import type {
    ZubeiaMatchedCrackComponent,
    ZubeiaMatchedLiftedCrustComponent,
} from "./ZubeiaMatchedTopologyProof";
import {
    POSITION_QUANTIZATION_EPSILON,
    type CachedFrameView,
    componentAt,
    maskIndexForCell,
    maskIndexForPosition,
    materialAt,
    positionAt,
} from "./ZubeiaMatchedTopologyFrame";

export const ZUBEIA_MATCHED_LIFT_MIN_HEIGHT_CELLS = 0.5;
export const ZUBEIA_MATCHED_CRACK_MIN_OPENING_CELLS = 1;

type LiftedCell = {
    x: number,
    y: number,
    z: number,
    particleIndices: number[],
    positionSum: [number, number, number],
    liftSumM: number,
    maxLiftM: number,
    boundsMinM: [number, number, number],
    boundsMaxM: [number, number, number],
};

type BrokenEdge = {
    a: number,
    b: number,
    matchedOpeningM: number,
    cellIndices: readonly number[],
};

const cellKey3 = (x: number, y: number, z: number) =>
    `${x},${y},${z}`;

const addLiftedParticle = ({
    cells,
    particleIndex,
    position,
    liftM,
    gridCellSizeM,
}: {
    cells: Map<string, LiftedCell>,
    particleIndex: number,
    position: readonly [number, number, number],
    liftM: number,
    gridCellSizeM: number,
}) => {
    const x = Math.floor(position[0] / gridCellSizeM);
    const y = Math.floor(position[1] / gridCellSizeM);
    const z = Math.floor(position[2] / gridCellSizeM);
    const key = cellKey3(x, y, z);
    let cell = cells.get(key);
    if (cell === undefined) {
        cell = {
            x,
            y,
            z,
            particleIndices: [],
            positionSum: [0, 0, 0],
            liftSumM: 0,
            maxLiftM: 0,
            boundsMinM: [...position],
            boundsMaxM: [...position],
        };
        cells.set(key, cell);
    }

    cell.particleIndices.push(particleIndex);
    cell.positionSum[0] += position[0];
    cell.positionSum[1] += position[1];
    cell.positionSum[2] += position[2];
    cell.liftSumM += liftM;
    cell.maxLiftM = Math.max(cell.maxLiftM, liftM);
    for (let axis = 0; axis < 3; axis++) {
        cell.boundsMinM[axis] = Math.min(cell.boundsMinM[axis], position[axis]);
        cell.boundsMaxM[axis] = Math.max(cell.boundsMaxM[axis], position[axis]);
    }
};

const liftedComponentsFromCells = (
    cells: Map<string, LiftedCell>,
) => {
    const visited = new Set<string>();
    const components: ZubeiaMatchedLiftedCrustComponent[] = [];

    for (const [startKey, startCell] of cells) {
        if (visited.has(startKey)) continue;

        const queue = [startCell];
        const particleIndices: number[] = [];
        const positionSum = [0, 0, 0];
        let liftSumM = 0;
        let maxLiftM = 0;
        const boundsMinM = [Infinity, Infinity, Infinity];
        const boundsMaxM = [-Infinity, -Infinity, -Infinity];
        visited.add(startKey);

        for (let queueIndex = 0; queueIndex < queue.length; queueIndex++) {
            const cell = queue[queueIndex];
            particleIndices.push(...cell.particleIndices);
            liftSumM += cell.liftSumM;
            maxLiftM = Math.max(maxLiftM, cell.maxLiftM);
            for (let axis = 0; axis < 3; axis++) {
                positionSum[axis] += cell.positionSum[axis];
                boundsMinM[axis] = Math.min(boundsMinM[axis], cell.boundsMinM[axis]);
                boundsMaxM[axis] = Math.max(boundsMaxM[axis], cell.boundsMaxM[axis]);
            }

            for (let dz = -1; dz <= 1; dz++) {
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        if (dx === 0 && dy === 0 && dz === 0) continue;

                        const neighborKey = cellKey3(
                            cell.x + dx,
                            cell.y + dy,
                            cell.z + dz,
                        );
                        if (visited.has(neighborKey)) continue;

                        const neighbor = cells.get(neighborKey);
                        if (neighbor === undefined) continue;

                        visited.add(neighborKey);
                        queue.push(neighbor);
                    }
                }
            }
        }

        particleIndices.sort((a, b) => a - b);
        const particleCount = particleIndices.length;
        components.push({
            particleCount,
            particleIndices,
            centroidM: [
                positionSum[0] / particleCount,
                positionSum[1] / particleCount,
                positionSum[2] / particleCount,
            ],
            centroidLiftM: liftSumM / particleCount,
            maxLiftM,
            boundsMinM: [boundsMinM[0], boundsMinM[1], boundsMinM[2]],
            boundsMaxM: [boundsMaxM[0], boundsMaxM[1], boundsMaxM[2]],
        });
    }

    return components.sort((a, b) =>
        b.particleCount - a.particleCount
        || a.particleIndices[0] - b.particleIndices[0]
    );
};

export const buildMatchedLiftedCrustComponents = ({
    colliderBaseline,
    colliderCurrent,
    particleCount,
    residualZ,
    residualCenterM,
    gridCellSizeM,
}: {
    colliderBaseline: CachedFrameView,
    colliderCurrent: CachedFrameView,
    particleCount: number,
    residualZ: Float32Array,
    residualCenterM: number,
    gridCellSizeM: number,
}) => {
    const cells = new Map<string, LiftedCell>();
    const minimumLiftM =
        ZUBEIA_MATCHED_LIFT_MIN_HEIGHT_CELLS * gridCellSizeM;

    for (let particleIndex = 0; particleIndex < particleCount; particleIndex++) {
        if (
            materialAt(colliderBaseline, particleIndex)
            !== ZUBEIA_GROUND_CRUST_MATERIAL_ID
        ) {
            continue;
        }

        const liftM = residualZ[particleIndex] - residualCenterM;
        if (liftM < minimumLiftM) continue;

        addLiftedParticle({
            cells,
            particleIndex,
            position: positionAt(colliderCurrent, particleIndex),
            liftM,
            gridCellSizeM,
        });
    }

    return liftedComponentsFromCells(cells);
};

const xyDistance = (
    frame: CachedFrameView,
    a: number,
    b: number,
) => Math.hypot(
    componentAt(frame, a, 0) - componentAt(frame, b, 0),
    componentAt(frame, a, 1) - componentAt(frame, b, 1),
);

const edgeCellIndices = (
    baseline: CachedFrameView,
    a: number,
    b: number,
    mask: ZubeiaPawContactMask,
) => {
    const indices = new Set<number>();
    const points = [
        [componentAt(baseline, a, 0), componentAt(baseline, a, 1)],
        [componentAt(baseline, b, 0), componentAt(baseline, b, 1)],
        [
            (componentAt(baseline, a, 0) + componentAt(baseline, b, 0)) / 2,
            (componentAt(baseline, a, 1) + componentAt(baseline, b, 1)) / 2,
        ],
    ];
    for (const [xM, yM] of points) {
        const cellIndex = maskIndexForPosition(xM, yM, mask);
        if (cellIndex >= 0) indices.add(cellIndex);
    }

    return [...indices].sort((left, right) => left - right);
};

const crackComponentsFromEdges = ({
    edges,
    baseline,
    perimeter,
}: {
    edges: readonly BrokenEdge[],
    baseline: CachedFrameView,
    perimeter: Uint8Array,
}) => {
    const edgeIndicesByParticle = new Map<number, number[]>();
    edges.forEach((edge, edgeIndex) => {
        for (const particleIndex of [edge.a, edge.b]) {
            const edgeIndices = edgeIndicesByParticle.get(particleIndex) ?? [];
            edgeIndices.push(edgeIndex);
            edgeIndicesByParticle.set(particleIndex, edgeIndices);
        }
    });

    const visited = new Uint8Array(edges.length);
    const components: ZubeiaMatchedCrackComponent[] = [];
    for (let startEdgeIndex = 0; startEdgeIndex < edges.length; startEdgeIndex++) {
        if (visited[startEdgeIndex] !== 0) continue;

        const queue = [startEdgeIndex];
        const particleIndices = new Set<number>();
        const cellIndices = new Set<number>();
        let openingSumM = 0;
        let maxMatchedOpeningM = 0;
        visited[startEdgeIndex] = 1;

        for (let queueIndex = 0; queueIndex < queue.length; queueIndex++) {
            const edge = edges[queue[queueIndex]];
            openingSumM += edge.matchedOpeningM;
            maxMatchedOpeningM = Math.max(
                maxMatchedOpeningM,
                edge.matchedOpeningM,
            );
            for (const cellIndex of edge.cellIndices) cellIndices.add(cellIndex);

            for (const particleIndex of [edge.a, edge.b]) {
                particleIndices.add(particleIndex);
                for (
                    const neighborEdgeIndex
                    of edgeIndicesByParticle.get(particleIndex) ?? []
                ) {
                    if (visited[neighborEdgeIndex] !== 0) continue;

                    visited[neighborEdgeIndex] = 1;
                    queue.push(neighborEdgeIndex);
                }
            }
        }

        const sortedParticleIndices = [...particleIndices].sort((a, b) => a - b);
        const sortedCellIndices = [...cellIndices].sort((a, b) => a - b);
        const boundsMinM = [Infinity, Infinity];
        const boundsMaxM = [-Infinity, -Infinity];
        for (const particleIndex of sortedParticleIndices) {
            for (let axis = 0; axis < 2; axis++) {
                const value = componentAt(baseline, particleIndex, axis);
                boundsMinM[axis] = Math.min(boundsMinM[axis], value);
                boundsMaxM[axis] = Math.max(boundsMaxM[axis], value);
            }
        }

        components.push({
            edgeCount: queue.length,
            particleCount: sortedParticleIndices.length,
            particleIndices: sortedParticleIndices,
            cellIndices: sortedCellIndices,
            meanMatchedOpeningM: openingSumM / queue.length,
            maxMatchedOpeningM,
            boundsMinM: [boundsMinM[0], boundsMinM[1]],
            boundsMaxM: [boundsMaxM[0], boundsMaxM[1]],
            intersectsPawPerimeter: sortedCellIndices.some(
                (cellIndex) => perimeter[cellIndex] !== 0,
            ),
        });
    }

    return components.sort((a, b) =>
        b.edgeCount - a.edgeCount
        || a.particleIndices[0] - b.particleIndices[0]
    );
};

export const analyzeMatchedCracks = ({
    colliderBaseline,
    colliderCurrent,
    controlCurrent,
    particleCount,
    topCrustParticleMask,
    mask,
    gridCellSizeM,
}: {
    colliderBaseline: CachedFrameView,
    colliderCurrent: CachedFrameView,
    controlCurrent: CachedFrameView,
    particleCount: number,
    topCrustParticleMask: Uint8Array,
    mask: ZubeiaPawContactMask,
    gridCellSizeM: number,
}) => {
    const cellCount = mask.walkingCellCount * mask.crossStripCellCount;
    const particlesByCell = new Map<number, number[]>();

    for (let particleIndex = 0; particleIndex < particleCount; particleIndex++) {
        if (topCrustParticleMask[particleIndex] === 0) continue;

        const cellIndex = maskIndexForPosition(
            componentAt(colliderBaseline, particleIndex, 0),
            componentAt(colliderBaseline, particleIndex, 1),
            mask,
        );
        const particles = particlesByCell.get(cellIndex) ?? [];
        particles.push(particleIndex);
        particlesByCell.set(cellIndex, particles);
    }

    const openingThresholdM =
        ZUBEIA_MATCHED_CRACK_MIN_OPENING_CELLS * gridCellSizeM;
    const thresholdToleranceM = gridCellSizeM * 1e-5;
    const maximumBaselineEdgeLengthM =
        gridCellSizeM * (1 + POSITION_QUANTIZATION_EPSILON);
    const validBrokenEdges: BrokenEdge[] = [];
    const controlOpeningCellMask = new Uint8Array(cellCount);
    let matchedBrokenEdgeCount = 0;
    let controlOpeningEdgeCount = 0;
    let controlInvalidatedBrokenEdgeCount = 0;

    for (const [cellIndex, particles] of particlesByCell) {
        const x = Math.floor(cellIndex / mask.walkingCellCount);
        const y = cellIndex % mask.walkingCellCount;
        for (const a of particles) {
            for (let dx = -1; dx <= 1; dx++) {
                for (let dy = -1; dy <= 1; dy++) {
                    const neighborCellIndex = maskIndexForCell(
                        x + dx,
                        y + dy,
                        mask,
                    );
                    if (neighborCellIndex < 0) continue;

                    for (const b of particlesByCell.get(neighborCellIndex) ?? []) {
                        if (b <= a) continue;

                        const baselineDistanceM = xyDistance(
                            colliderBaseline,
                            a,
                            b,
                        );
                        if (
                            baselineDistanceM <= thresholdToleranceM
                            || baselineDistanceM > maximumBaselineEdgeLengthM
                        ) {
                            continue;
                        }

                        const colliderDistanceM = xyDistance(
                            colliderCurrent,
                            a,
                            b,
                        );
                        const controlDistanceM = xyDistance(
                            controlCurrent,
                            a,
                            b,
                        );
                        const matchedOpeningM =
                            colliderDistanceM - controlDistanceM;
                        const controlOpeningM =
                            controlDistanceM - baselineDistanceM;
                        const isMatchedBroken =
                            matchedOpeningM + thresholdToleranceM
                            >= openingThresholdM;
                        const isControlOpening =
                            controlOpeningM + thresholdToleranceM
                            >= openingThresholdM;
                        if (!isMatchedBroken && !isControlOpening) continue;

                        const cellIndices = edgeCellIndices(
                            colliderBaseline,
                            a,
                            b,
                            mask,
                        );
                        if (isControlOpening) {
                            controlOpeningEdgeCount++;
                            for (const index of cellIndices) {
                                controlOpeningCellMask[index] = 1;
                            }
                        }
                        if (!isMatchedBroken) continue;

                        matchedBrokenEdgeCount++;
                        if (isControlOpening) {
                            controlInvalidatedBrokenEdgeCount++;
                            continue;
                        }

                        validBrokenEdges.push({
                            a,
                            b,
                            matchedOpeningM,
                            cellIndices,
                        });
                    }
                }
            }
        }
    }

    const crackComponents = crackComponentsFromEdges({
        edges: validBrokenEdges,
        baseline: colliderBaseline,
        perimeter: mask.perimeter,
    });
    const qualifyingCrackComponents = crackComponents.filter(
        (component) => component.intersectsPawPerimeter,
    );
    const crackCellMask = new Uint8Array(cellCount);
    for (const component of qualifyingCrackComponents) {
        for (const cellIndex of component.cellIndices) {
            crackCellMask[cellIndex] = 1;
        }
    }

    return {
        matchedBrokenEdgeCount,
        controlOpeningEdgeCount,
        controlInvalidatedBrokenEdgeCount,
        crackComponents,
        qualifyingCrackComponents,
        crackCellMask,
        controlOpeningCellMask,
    };
};
