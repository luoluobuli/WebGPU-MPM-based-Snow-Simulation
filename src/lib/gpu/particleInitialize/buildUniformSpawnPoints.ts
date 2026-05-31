type Vec3 = [number, number, number];

export type SpawnMeshObject = {
    min?: Vec3;
    max?: Vec3;
    startVertex: number;
    countVertices: number;
};

export type UniformSpawnPoints = {
    points: Float32Array;
    pointCount: number;
    candidateCount: number;
    spacing: number;
};

type SpawnComponent = {
    min: Vec3;
    max: Vec3;
    startVertex: number;
    countVertices: number;
};

const CANDIDATE_OVERSAMPLE = 2;
const INSIDE_EPSILON = 1e-5;
const MAX_ADAPTIVE_ATTEMPTS = 6;
const CELL_RANDOM_MARGIN = 0.48;

const add = (a: Vec3, b: Vec3): Vec3 => [
    a[0] + b[0],
    a[1] + b[1],
    a[2] + b[2],
];

const subtract = (a: Vec3, b: Vec3): Vec3 => [
    a[0] - b[0],
    a[1] - b[1],
    a[2] - b[2],
];

const cross = (a: Vec3, b: Vec3): Vec3 => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
];

const dot = (a: Vec3, b: Vec3) =>
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

const hashUint = (value: number): number => {
    let x = value | 0;
    x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
    x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
    return (x ^ (x >>> 16)) >>> 0;
};

const hashInts = (...values: number[]) => {
    let hash = 0x811c9dc5;

    for (const value of values) {
        hash = Math.imul(hash ^ (value | 0), 0x01000193);
    }

    return hashUint(hash);
};

const jitter = (...values: number[]) =>
    (hashInts(...values) / 0xffffffff - 0.5) * 2;

const rand01 = (...values: number[]) =>
    hashInts(...values) / 0xffffffff;

const triangleSignedVolume = (v0: Vec3, v1: Vec3, v2: Vec3) =>
    dot(v0, cross(v1, v2)) / 6;

const triangleFanVolume = (
    vertices: number[][],
    component: SpawnComponent,
) => {
    const center: Vec3 = [0, 0, 0];
    const end = component.startVertex + component.countVertices;

    for (let i = component.startVertex; i < end; i++) {
        center[0] += vertices[i][0];
        center[1] += vertices[i][1];
        center[2] += vertices[i][2];
    }

    const invCount = 1 / component.countVertices;
    center[0] *= invCount;
    center[1] *= invCount;
    center[2] *= invCount;

    let volume = 0;
    for (let i = component.startVertex; i < end; i += 3) {
        const v0 = subtract(vertices[i] as Vec3, center);
        const v1 = subtract(vertices[i + 1] as Vec3, center);
        const v2 = subtract(vertices[i + 2] as Vec3, center);
        volume += Math.abs(dot(v0, cross(v1, v2)) / 6);
    }

    return volume;
};

const componentBounds = (
    vertices: number[][],
    component: SpawnMeshObject,
): { min: Vec3, max: Vec3 } => {
    const min: Vec3 = component.min ? [...component.min] : [Infinity, Infinity, Infinity];
    const max: Vec3 = component.max ? [...component.max] : [-Infinity, -Infinity, -Infinity];
    const end = component.startVertex + component.countVertices;

    if (component.min !== undefined && component.max !== undefined) {
        return { min, max };
    }

    for (let i = component.startVertex; i < end; i++) {
        const vertex = vertices[i];
        min[0] = Math.min(min[0], vertex[0]);
        min[1] = Math.min(min[1], vertex[1]);
        min[2] = Math.min(min[2], vertex[2]);
        max[0] = Math.max(max[0], vertex[0]);
        max[1] = Math.max(max[1], vertex[1]);
        max[2] = Math.max(max[2], vertex[2]);
    }

    return { min, max };
};

const buildComponents = (
    vertices: number[][],
    objects?: SpawnMeshObject[],
) => {
    const sourceComponents = objects?.filter((object) => object.countVertices >= 3) ?? [
        {
            startVertex: 0,
            countVertices: vertices.length,
        },
    ];

    const components: SpawnComponent[] = [];

    for (const component of sourceComponents) {
        const end = component.startVertex + component.countVertices;
        if (component.startVertex < 0 || end > vertices.length || component.countVertices % 3 !== 0) {
            continue;
        }

        const { min, max } = componentBounds(vertices, component);
        if (
            !min.every(Number.isFinite)
            || !max.every(Number.isFinite)
            || max[0] <= min[0]
            || max[1] <= min[1]
            || max[2] <= min[2]
        ) {
            continue;
        }

        components.push({
            min,
            max,
            startVertex: component.startVertex,
            countVertices: component.countVertices,
        });
    }

    return components;
};

const projectedBarycentricYZ = (
    y: number,
    z: number,
    v0: number[],
    v1: number[],
    v2: number[],
): Vec3 | null => {
    const y0 = v0[1];
    const z0 = v0[2];
    const y1 = v1[1];
    const z1 = v1[2];
    const y2 = v2[1];
    const z2 = v2[2];
    const denom = (z1 - z2) * (y0 - y2) + (y2 - y1) * (z0 - z2);

    if (Math.abs(denom) < 1e-12) return null;

    const w0 = ((z1 - z2) * (y - y2) + (y2 - y1) * (z - z2)) / denom;
    const w1 = ((z2 - z0) * (y - y2) + (y0 - y2) * (z - z2)) / denom;
    const w2 = 1 - w0 - w1;

    if (w0 < -INSIDE_EPSILON || w1 < -INSIDE_EPSILON || w2 < -INSIDE_EPSILON) {
        return null;
    }

    return [w0, w1, w2];
};

const estimateVolume = (
    vertices: number[][],
    components: SpawnComponent[],
) => {
    let signedVolume = 0;
    let fanVolume = 0;

    for (const component of components) {
        let componentSignedVolume = 0;
        const end = component.startVertex + component.countVertices;

        for (let i = component.startVertex; i < end; i += 3) {
            componentSignedVolume += triangleSignedVolume(
                vertices[i] as Vec3,
                vertices[i + 1] as Vec3,
                vertices[i + 2] as Vec3,
            );
        }

        signedVolume += Math.abs(componentSignedVolume);
        fanVolume += triangleFanVolume(vertices, component);
    }

    return signedVolume > 1e-8 ? signedVolume : fanVolume;
};

const boundsVolume = (components: SpawnComponent[]) => {
    let volume = 0;

    for (const component of components) {
        volume += (component.max[0] - component.min[0])
            * (component.max[1] - component.min[1])
            * (component.max[2] - component.min[2]);
    }

    return volume;
};

const buildCandidatePoints = (
    vertices: number[][],
    components: SpawnComponent[],
    spacing: number,
) => {
    const candidates: number[] = [];
    const crossingMergeEpsilon = Math.max(1e-5, spacing * 1e-4);
    const intervalEpsilon = Math.max(1e-5, spacing * 0.05);

    for (let componentIndex = 0; componentIndex < components.length; componentIndex++) {
        const component = components[componentIndex];
        const min = subtract(component.min, [spacing, spacing, spacing]);
        const max = add(component.max, [spacing, spacing, spacing]);
        const ny = Math.max(1, Math.ceil((max[1] - min[1]) / spacing));
        const nz = Math.max(1, Math.ceil((max[2] - min[2]) / spacing));
        const cellCount = ny * nz;
        const crossings = Array.from({ length: cellCount }, () => [] as number[]);
        const scanY = new Float32Array(cellCount);
        const scanZ = new Float32Array(cellCount);

        for (let iy = 0; iy < ny; iy++) {
            for (let iz = 0; iz < nz; iz++) {
                const cellIndex = iy * nz + iz;
                scanY[cellIndex] = min[1] + (iy + 0.5) * spacing;
                scanZ[cellIndex] = min[2] + (iz + 0.5) * spacing;
            }
        }

        const end = component.startVertex + component.countVertices;
        for (let i = component.startVertex; i < end; i += 3) {
            const v0 = vertices[i];
            const v1 = vertices[i + 1];
            const v2 = vertices[i + 2];
            const triMinY = Math.min(v0[1], v1[1], v2[1]);
            const triMaxY = Math.max(v0[1], v1[1], v2[1]);
            const triMinZ = Math.min(v0[2], v1[2], v2[2]);
            const triMaxZ = Math.max(v0[2], v1[2], v2[2]);
            const iy0 = Math.max(0, Math.floor((triMinY - min[1]) / spacing) - 1);
            const iy1 = Math.min(ny - 1, Math.ceil((triMaxY - min[1]) / spacing) + 1);
            const iz0 = Math.max(0, Math.floor((triMinZ - min[2]) / spacing) - 1);
            const iz1 = Math.min(nz - 1, Math.ceil((triMaxZ - min[2]) / spacing) + 1);

            for (let iy = iy0; iy <= iy1; iy++) {
                for (let iz = iz0; iz <= iz1; iz++) {
                    const cellIndex = iy * nz + iz;
                    const barycentric = projectedBarycentricYZ(scanY[cellIndex], scanZ[cellIndex], v0, v1, v2);
                    if (barycentric === null) continue;

                    crossings[cellIndex].push(
                        barycentric[0] * v0[0]
                        + barycentric[1] * v1[0]
                        + barycentric[2] * v2[0],
                    );
                }
            }
        }

        for (let cellIndex = 0; cellIndex < cellCount; cellIndex++) {
            const lineCrossings = crossings[cellIndex];
            if (lineCrossings.length < 2) continue;

            lineCrossings.sort((a, b) => a - b);

            const mergedCrossings: number[] = [];
            for (const x of lineCrossings) {
                const last = mergedCrossings[mergedCrossings.length - 1];
                if (last === undefined || Math.abs(x - last) > crossingMergeEpsilon) {
                    mergedCrossings.push(x);
                } else {
                    mergedCrossings[mergedCrossings.length - 1] = (last + x) * 0.5;
                }
            }

            for (let i = 0; i + 1 < mergedCrossings.length; i += 2) {
                const intervalMinX = mergedCrossings[i];
                const intervalMaxX = mergedCrossings[i + 1];
                if (intervalMaxX - intervalMinX <= intervalEpsilon) continue;

                const ix0 = Math.floor((intervalMinX - min[0]) / spacing);
                const ix1 = Math.floor((intervalMaxX - min[0]) / spacing);
                const iy = Math.floor(cellIndex / nz);
                const iz = cellIndex - iy * nz;

                for (let ix = ix0; ix <= ix1; ix++) {
                    const cellMinX = min[0] + ix * spacing;
                    const cellMaxX = cellMinX + spacing;
                    const sampleMinX = Math.max(cellMinX, intervalMinX + intervalEpsilon);
                    const sampleMaxX = Math.min(cellMaxX, intervalMaxX - intervalEpsilon);

                    if (sampleMaxX <= sampleMinX) {
                        continue;
                    }

                    const x = sampleMinX
                        + rand01(componentIndex, ix, cellIndex, 41) * (sampleMaxX - sampleMinX);
                    const y = min[1]
                        + (iy + 0.5 + jitter(componentIndex, ix, iy, iz, 53) * CELL_RANDOM_MARGIN) * spacing;
                    const z = min[2]
                        + (iz + 0.5 + jitter(componentIndex, ix, iy, iz, 67) * CELL_RANDOM_MARGIN) * spacing;

                    candidates.push(x, y, z, 0);
                }
            }
        }
    }

    return candidates;
};

const swapCandidate = (
    candidates: number[],
    a: number,
    b: number,
) => {
    if (a === b) return;

    const ai = a * 4;
    const bi = b * 4;

    for (let i = 0; i < 4; i++) {
        const tmp = candidates[ai + i];
        candidates[ai + i] = candidates[bi + i];
        candidates[bi + i] = tmp;
    }
};

const selectSpawnPoints = (
    candidates: number[],
    nPoints: number,
) => {
    const candidateCount = candidates.length / 4;
    const points = new Float32Array(nPoints * 4);

    if (candidateCount === 0) {
        throw new Error("uniform sampler produced no spawn points");
    }

    if (candidateCount >= nPoints) {
        for (let i = 0; i < nPoints; i++) {
            const remaining = candidateCount - i;
            const chosen = i + (hashInts(i, 0xdecafbad) % remaining);
            swapCandidate(candidates, i, chosen);
            points.set(candidates.slice(i * 4, i * 4 + 4), i * 4);
        }

        return points;
    }

    for (let i = 0; i < nPoints; i++) {
        const source = i < candidateCount
            ? i
            : hashInts(i, 0x9e3779b9) % candidateCount;
        const srcOffset = source * 4;
        points.set(candidates.slice(srcOffset, srcOffset + 4), i * 4);
    }

    return points;
};

export const buildUniformSpawnPoints = (
    vertices: number[][],
    {
        nPoints,
        objects,
    }: {
        nPoints: number,
        objects?: SpawnMeshObject[],
    },
): UniformSpawnPoints => {
    if (vertices.length < 3 || vertices.length % 3 !== 0) {
        throw new Error("spawn mesh must contain complete triangles");
    }

    if (!Number.isInteger(nPoints) || nPoints <= 0) {
        throw new Error("uniform sampler requires a positive point count");
    }

    const components = buildComponents(vertices, objects);
    if (components.length === 0) {
        throw new Error("uniform sampler found no valid mesh components");
    }

    const volume = estimateVolume(vertices, components);
    const fallbackVolume = boundsVolume(components);
    const estimatedVolume = Number.isFinite(volume) && volume > 1e-8
        ? volume
        : Math.max(fallbackVolume * 0.25, 1e-8);
    let spacing = Math.cbrt(estimatedVolume / (nPoints * CANDIDATE_OVERSAMPLE));
    let candidates: number[] = [];

    for (let attempt = 0; attempt < MAX_ADAPTIVE_ATTEMPTS; attempt++) {
        candidates = buildCandidatePoints(vertices, components, spacing);
        const candidateCount = candidates.length / 4;

        if (candidateCount >= nPoints || attempt === MAX_ADAPTIVE_ATTEMPTS - 1) {
            break;
        }

        spacing *= candidateCount === 0
            ? 0.5
            : Math.max(0.45, Math.min(0.85, Math.cbrt(candidateCount / nPoints) * 0.9));
    }

    return {
        points: selectSpawnPoints(candidates, nPoints),
        pointCount: nPoints,
        candidateCount: candidates.length / 4,
        spacing,
    };
};
