import type {
    PawSurfaceReference,
} from "./zubeiaKeyedSdfTypes.ts";

export const collectPawRegionReferences = (
    meshIndices: ReadonlyMap<string, Uint32Array>,
) => {
    const result: PawSurfaceReference[] = [];
    meshIndices.forEach((indices, meshName) => {
        const vertices = new Set<number>();
        indices.forEach(vertexIndex => vertices.add(vertexIndex));
        vertices.forEach(vertexIndex => {
            result.push({
                meshName,
                vertexIndex,
            });
        });
    });

    return result;
};
