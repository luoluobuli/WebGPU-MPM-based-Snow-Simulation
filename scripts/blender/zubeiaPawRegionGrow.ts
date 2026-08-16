import type {
    MeshFrame,
} from "./zubeiaKeyedSdfTypes.ts";

type EdgeRecord = {
    firstVertex: number,
    secondVertex: number,
    triangles: number[],
};

const triangleKey = (
    first: number,
    second: number,
    third: number,
) => [first, second, third].sort((a, b) => a - b).join(",");

const edgeKey = (first: number, second: number) =>
    first < second ? `${first},${second}` : `${second},${first}`;

const minimumAnimatedEdgeZ = (
    frames: readonly ReadonlyMap<string, MeshFrame>[],
    meshName: string,
    firstVertex: number,
    secondVertex: number,
) => {
    let result = Infinity;
    frames.forEach(frame => {
        const positions = frame.get(meshName)!.positions;
        result = Math.min(
            result,
            positions[firstVertex * 3 + 2],
            positions[secondVertex * 3 + 2],
        );
    });
    return result;
};

export const growPawRegionsAboveContactBand = ({
    frames,
    initialRegionIndices,
    minimumClosureZM,
}: {
    frames: readonly ReadonlyMap<string, MeshFrame>[],
    initialRegionIndices: ReadonlyMap<string, Uint32Array>,
    minimumClosureZM: number,
}) => {
    const result = new Map<string, Uint32Array>();

    initialRegionIndices.forEach((initialIndices, meshName) => {
        const fullIndices = frames[0].get(meshName)!.indices;
        const triangleCount = fullIndices.length / 3;
        const selected = new Uint8Array(triangleCount);
        const selectedKeys = new Set<string>();
        for (
            let triangleOffset = 0;
            triangleOffset < initialIndices.length;
            triangleOffset += 3
        ) {
            selectedKeys.add(triangleKey(
                initialIndices[triangleOffset],
                initialIndices[triangleOffset + 1],
                initialIndices[triangleOffset + 2],
            ));
        }

        const edges = new Map<string, EdgeRecord>();
        for (
            let triangleIndex = 0;
            triangleIndex < triangleCount;
            triangleIndex++
        ) {
            const triangleOffset = triangleIndex * 3;
            const first = fullIndices[triangleOffset];
            const second = fullIndices[triangleOffset + 1];
            const third = fullIndices[triangleOffset + 2];
            selected[triangleIndex] = Number(selectedKeys.has(
                triangleKey(first, second, third),
            ));
            const vertices = [first, second, third];
            for (let edge = 0; edge < 3; edge++) {
                const firstVertex = vertices[edge];
                const secondVertex = vertices[(edge + 1) % 3];
                const key = edgeKey(firstVertex, secondVertex);
                let record = edges.get(key);
                if (record === undefined) {
                    record = {
                        firstVertex,
                        secondVertex,
                        triangles: [],
                    };
                    edges.set(key, record);
                }
                record.triangles.push(triangleIndex);
            }
        }

        let changed = true;
        let iteration = 0;
        while (changed) {
            changed = false;
            iteration++;
            if (iteration > triangleCount) {
                throw new Error(`${meshName} paw region growth did not end`);
            }

            const additions = new Set<number>();
            edges.forEach(edge => {
                if (edge.triangles.length !== 2) return;
                const firstSelected = selected[edge.triangles[0]] !== 0;
                const secondSelected = selected[edge.triangles[1]] !== 0;
                if (firstSelected === secondSelected) return;
                if (
                    minimumAnimatedEdgeZ(
                        frames,
                        meshName,
                        edge.firstVertex,
                        edge.secondVertex,
                    ) >= minimumClosureZM
                ) {
                    return;
                }
                additions.add(
                    firstSelected
                        ? edge.triangles[1]
                        : edge.triangles[0],
                );
            });

            additions.forEach(triangleIndex => {
                if (selected[triangleIndex] === 0) {
                    selected[triangleIndex] = 1;
                    changed = true;
                }
            });
        }

        const grownIndices: number[] = [];
        selected.forEach((isSelected, triangleIndex) => {
            if (isSelected === 0) return;
            const triangleOffset = triangleIndex * 3;
            grownIndices.push(
                fullIndices[triangleOffset],
                fullIndices[triangleOffset + 1],
                fullIndices[triangleOffset + 2],
            );
        });
        result.set(meshName, Uint32Array.from(grownIndices));
    });

    return result;
};
