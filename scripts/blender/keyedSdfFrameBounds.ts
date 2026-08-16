import type { Vec3 } from "./keyedSdfBvh.ts";
import type {
    MeshFrame,
    PawFieldBounds,
    PawSource,
} from "./zubeiaKeyedSdfTypes.ts";

export const measureTightPawFieldBounds = ({
    frame,
    paw,
    paddingM,
}: {
    frame: ReadonlyMap<string, MeshFrame>,
    paw: PawSource,
    paddingM: number,
}): PawFieldBounds => {
    const minLocal: Vec3 = [Infinity, Infinity, Infinity];
    const maxLocal: Vec3 = [-Infinity, -Infinity, -Infinity];

    paw.meshIndices.forEach((indices, meshName) => {
        const positions = frame.get(meshName)!.positions;
        const uniqueVertexIndices = new Set(indices);
        uniqueVertexIndices.forEach(vertexIndex => {
            for (let axis = 0; axis < 3; axis++) {
                const value = positions[vertexIndex * 3 + axis];
                minLocal[axis] = Math.min(minLocal[axis], value);
                maxLocal[axis] = Math.max(maxLocal[axis], value);
            }
        });
    });

    if (
        minLocal.some(value => !Number.isFinite(value))
        || maxLocal.some(value => !Number.isFinite(value))
    ) {
        throw new Error("cannot measure an empty paw field");
    }

    return {
        minLocal: minLocal.map(value => value - paddingM) as Vec3,
        maxLocal: maxLocal.map(value => value + paddingM) as Vec3,
    };
};
