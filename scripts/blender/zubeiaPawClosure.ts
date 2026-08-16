import type {
    MeshFrame,
} from "./zubeiaKeyedSdfTypes.ts";

export type PawMeshClosure = {
    indices: Uint32Array,
    capLoops: readonly (readonly number[])[],
};

const readPositionKey = (
    positions: Float32Array,
    vertexIndex: number,
    weldToleranceM: number,
) => {
    const offset = vertexIndex * 3;

    return [
        positions[offset],
        positions[offset + 1],
        positions[offset + 2],
    ].map(
        value => Math.round(value / weldToleranceM),
    ).join(",");
};

export const closeWeightedPawRegions = ({
    frame,
    regionIndices,
    weldToleranceM,
}: {
    frame: ReadonlyMap<string, MeshFrame>,
    regionIndices: ReadonlyMap<string, Uint32Array>,
    weldToleranceM: number,
}) => {
    const result = new Map<string, PawMeshClosure>();

    regionIndices.forEach((indices, meshName) => {
        const meshFrame = frame.get(meshName)!;
        const weldedByKey = new Map<string, number>();
        const weldedByVertex = new Map<number, number>();
        const representativeByWelded: number[] = [];

        indices.forEach(vertexIndex => {
            if (weldedByVertex.has(vertexIndex)) return;
            const key = readPositionKey(
                meshFrame.positions,
                vertexIndex,
                weldToleranceM,
            );
            let weldedIndex = weldedByKey.get(key);
            if (weldedIndex === undefined) {
                weldedIndex = representativeByWelded.length;
                weldedByKey.set(key, weldedIndex);
                representativeByWelded.push(vertexIndex);
            }
            weldedByVertex.set(vertexIndex, weldedIndex);
        });

        const edges = new Map<
            string,
            {
                count: number,
                from: number,
                to: number,
            }
        >();
        for (
            let triangleOffset = 0;
            triangleOffset < indices.length;
            triangleOffset += 3
        ) {
            for (let edge = 0; edge < 3; edge++) {
                const rawFrom = indices[triangleOffset + edge];
                const rawTo = indices[
                    triangleOffset + (edge + 1) % 3
                ];
                const from = weldedByVertex.get(rawFrom)!;
                const to = weldedByVertex.get(rawTo)!;
                if (from === to) continue;
                const key = from < to
                    ? `${from},${to}`
                    : `${to},${from}`;
                const existing = edges.get(key);
                if (existing === undefined) {
                    edges.set(key, {
                        count: 1,
                        from,
                        to,
                    });
                } else {
                    existing.count++;
                }
            }
        }

        const nextByVertex = new Map<number, number>();
        const previousCount = new Map<number, number>();
        edges.forEach(edge => {
            if (edge.count !== 1) return;
            if (nextByVertex.has(edge.from)) {
                throw new Error(
                    `${meshName} paw boundary branches at a vertex`,
                );
            }
            nextByVertex.set(edge.from, edge.to);
            previousCount.set(
                edge.to,
                (previousCount.get(edge.to) ?? 0) + 1,
            );
        });
        if (nextByVertex.size < 3) {
            throw new Error(`${meshName} paw region has no open boundary`);
        }
        nextByVertex.forEach((_, vertex) => {
            if ((previousCount.get(vertex) ?? 0) !== 1) {
                throw new Error(
                    `${meshName} paw boundary is not a simple loop`,
                );
            }
        });

        const unvisited = new Set(nextByVertex.keys());
        const capLoops: number[][] = [];
        while (unvisited.size > 0) {
            const first = unvisited.values().next().value as number;
            const weldedLoop: number[] = [];
            let current = first;
            do {
                if (!unvisited.delete(current)) {
                    throw new Error(
                        `${meshName} paw boundary revisits a loop vertex`,
                    );
                }
                weldedLoop.push(current);
                current = nextByVertex.get(current)!;
            } while (current !== first);
            if (weldedLoop.length < 3) {
                throw new Error(`${meshName} paw cap has too few vertices`);
            }
            capLoops.push(
                weldedLoop.map(
                    weldedIndex => representativeByWelded[weldedIndex],
                ),
            );
        }

        const closedIndices = [...indices];
        const sourceVertexCount = meshFrame.positions.length / 3;
        capLoops.forEach((loop, loopIndex) => {
            const centerIndex = sourceVertexCount + loopIndex;
            for (let index = 0; index < loop.length; index++) {
                closedIndices.push(
                    loop[(index + 1) % loop.length],
                    loop[index],
                    centerIndex,
                );
            }
        });
        result.set(meshName, {
            indices: Uint32Array.from(closedIndices),
            capLoops,
        });
    });

    return result;
};

export const applyPawMeshClosure = ({
    frame,
    closure,
}: {
    frame: MeshFrame,
    closure: PawMeshClosure,
}): MeshFrame => {
    const sourceVertexCount = frame.positions.length / 3;
    const positions = new Float32Array(
        frame.positions.length + closure.capLoops.length * 3,
    );
    positions.set(frame.positions);

    closure.capLoops.forEach((loop, loopIndex) => {
        const centerOffset = (sourceVertexCount + loopIndex) * 3;
        loop.forEach(vertexIndex => {
            const sourceOffset = vertexIndex * 3;
            positions[centerOffset] += frame.positions[sourceOffset]
                / loop.length;
            positions[centerOffset + 1] +=
                frame.positions[sourceOffset + 1] / loop.length;
            positions[centerOffset + 2] +=
                frame.positions[sourceOffset + 2] / loop.length;
        });
    });

    return {
        positions,
        indices: closure.indices,
    };
};
