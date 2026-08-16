export type Vec3 = [number, number, number];

export type NearestTriangle = {
    squaredDistance: number,
    triangleIndex: number,
    barycentric: Vec3,
};

const position = (
    positions: Float32Array,
    vertexIndex: number,
): Vec3 => {
    const offset = vertexIndex * 3;

    return [
        positions[offset],
        positions[offset + 1],
        positions[offset + 2],
    ];
};

export const validateWatertightTopology = ({
    positions,
    indices,
    weldToleranceM,
    minimumTriangleDoubleAreaM2 = 0,
}: {
    positions: Float32Array,
    indices: Uint32Array,
    weldToleranceM: number,
    minimumTriangleDoubleAreaM2?: number,
}) => {
    if (
        positions.length === 0
        || positions.length % 3 !== 0
        || positions.some(value => !Number.isFinite(value))
        || indices.length === 0
        || indices.length % 3 !== 0
        || !Number.isFinite(weldToleranceM)
        || weldToleranceM <= 0
        || !Number.isFinite(minimumTriangleDoubleAreaM2)
        || minimumTriangleDoubleAreaM2 < 0
    ) {
        throw new Error("mesh has invalid finite topology input");
    }

    const weldedIndexByPosition = new Map<string, number>();
    const weldedIndices = new Uint32Array(indices.length);
    indices.forEach((vertexIndex, indexOffset) => {
        if (vertexIndex * 3 + 2 >= positions.length) {
            throw new Error("mesh has an invalid vertex index");
        }
        const key = position(positions, vertexIndex)
            .map(value => Math.round(value / weldToleranceM))
            .join(",");
        let weldedIndex = weldedIndexByPosition.get(key);
        if (weldedIndex === undefined) {
            weldedIndex = weldedIndexByPosition.size;
            weldedIndexByPosition.set(key, weldedIndex);
        }
        weldedIndices[indexOffset] = weldedIndex;
    });

    const edgeUses = new Map<string, { balance: number, count: number }>();
    let minimumAreaM2 = Infinity;
    let signedVolumeM3 = 0;
    for (let offset = 0; offset < indices.length; offset += 3) {
        const welded = weldedIndices.slice(offset, offset + 3);
        if (new Set(welded).size !== 3) {
            throw new Error("mesh has a triangle that collapses after welding");
        }
        const a = position(positions, indices[offset]);
        const b = position(positions, indices[offset + 1]);
        const c = position(positions, indices[offset + 2]);
        const ab: Vec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        const ac: Vec3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        const areaVector: Vec3 = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        const doubleAreaM2 = Math.hypot(...areaVector);
        if (doubleAreaM2 < minimumTriangleDoubleAreaM2) {
            throw new Error("mesh has a degenerate triangle");
        }
        minimumAreaM2 = Math.min(minimumAreaM2, doubleAreaM2);
        signedVolumeM3 += (
            a[0] * areaVector[0]
            + a[1] * areaVector[1]
            + a[2] * areaVector[2]
        ) / 6;

        for (let edgeIndex = 0; edgeIndex < 3; edgeIndex++) {
            const from = welded[edgeIndex];
            const to = welded[(edgeIndex + 1) % 3];
            const key = from < to ? `${from},${to}` : `${to},${from}`;
            const edge = edgeUses.get(key) ?? { balance: 0, count: 0 };
            edge.balance += from < to ? 1 : -1;
            edge.count++;
            edgeUses.set(key, edge);
        }
    }

    const invalidEdgeCount = [...edgeUses.values()].filter(
        edge => edge.count !== 2 || edge.balance !== 0,
    ).length;
    if (invalidEdgeCount > 0) {
        throw new Error(
            `mesh has ${invalidEdgeCount} non-watertight or misdirected edges`,
        );
    }

    return {
        weldedVertexCount: weldedIndexByPosition.size,
        edgeCount: edgeUses.size,
        triangleCount: indices.length / 3,
        minimumTriangleDoubleAreaM2: minimumAreaM2,
        signedVolumeM3,
    };
};
