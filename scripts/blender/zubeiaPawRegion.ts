import type {
    MeshFrame,
    PawSurfaceReference,
} from "./zubeiaKeyedSdfTypes.ts";

export const createWeightedPawRegionIndices = ({
    frame,
    meshNames,
    references,
}: {
    frame: ReadonlyMap<string, MeshFrame>,
    meshNames: readonly string[],
    references: readonly PawSurfaceReference[],
}) => {
    const selectedVertices = new Map<string, Set<number>>();
    references.forEach(reference => {
        let selected = selectedVertices.get(reference.meshName);
        if (selected === undefined) {
            selected = new Set();
            selectedVertices.set(reference.meshName, selected);
        }
        selected.add(reference.vertexIndex);
    });

    const result = new Map<string, Uint32Array>();
    meshNames.forEach(meshName => {
        const meshFrame = frame.get(meshName)!;
        const selected = selectedVertices.get(meshName);
        if (selected === undefined || selected.size === 0) {
            throw new Error(`${meshName} has no weighted paw vertices`);
        }

        const regionIndices: number[] = [];
        for (
            let triangleOffset = 0;
            triangleOffset < meshFrame.indices.length;
            triangleOffset += 3
        ) {
            const first = meshFrame.indices[triangleOffset];
            const second = meshFrame.indices[triangleOffset + 1];
            const third = meshFrame.indices[triangleOffset + 2];
            if (
                selected.has(first)
                || selected.has(second)
                || selected.has(third)
            ) {
                regionIndices.push(first, second, third);
            }
        }
        if (regionIndices.length < 12) {
            throw new Error(
                `${meshName} paw region contains too few triangles`,
            );
        }
        result.set(meshName, Uint32Array.from(regionIndices));
    });

    return result;
};
