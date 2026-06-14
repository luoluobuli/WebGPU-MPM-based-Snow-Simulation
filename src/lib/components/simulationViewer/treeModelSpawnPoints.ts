import {
    ParticleAppearanceMaterial,
    packParticleAppearance,
} from "$lib/gpu/particleAppearance/GpuParticleAppearanceBufferManager";
import {
    buildUniformSpawnPoints,
    type SpawnMeshObject,
} from "$lib/gpu/particleInitialize/buildUniformSpawnPoints";
import {
    buildProceduralGround,
    DEFAULT_GROUND_BASE_Z,
} from "./proceduralForest";

export type Vec3 = [number, number, number];

export type TreeModelMesh = {
    name: string,
    vertices: Vec3[],
};

export type TreeModelSpawnPoints = {
    spawnPoints: Float32Array,
    particleAppearances: Uint32Array,
    groundCount: number,
    barkCount: number,
    leafCount: number,
    groundTopZ: number,
    groundBottomZ: number,
};

type Triangle = {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    normal: Vec3,
    cumulativeArea: number,
};

const DOMAIN_MIN: Vec3 = [-5, -5, -5];
const DOMAIN_MAX: Vec3 = [5, 5, 5];
const DOMAIN_PADDING = 0.22;
const SPAWN_CLAMP_PADDING = 0.08;
// The imported tree needs bark to remain the dominant support material;
// leaves are 2D sheets and should not consume the bulk of the tree budget.
const GROUND_PARTICLE_FRACTION = 0.18;
const LEAF_PARTICLE_FRACTION = 0.34;
export const TREE_MODEL_LEAF_SURFACE_THICKNESS = 0.045;
const BARK_BASE_COLOR: Vec3 = [0.43, 0.27, 0.14];
const LEAF_BASE_COLOR: Vec3 = [0.24, 0.48, 0.2];

type TreeModelFit = {
    meshes: TreeModelMesh[],
    groundTopZ: number,
    groundBottomZ: number,
    scale: number,
};

const clamp = (value: number, min: number, max: number) =>
    Math.max(min, Math.min(max, value));

const mix = (a: number, b: number, t: number) => a + (b - a) * t;

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

const length = (value: Vec3) =>
    Math.hypot(value[0], value[1], value[2]);

const normalize = (value: Vec3): Vec3 => {
    const len = length(value);

    return len > 1e-12
        ? [value[0] / len, value[1] / len, value[2] / len]
        : [0, 0, 1];
};

const triangleNormal = (
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) => normalize(cross(subtract(v1, v0), subtract(v2, v0)));

const triangleArea = (
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) => length(cross(subtract(v1, v0), subtract(v2, v0))) * 0.5;

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

const rand01 = (...values: number[]) =>
    hashInts(...values) / 0xffffffff;

const colorVariation = (
    base: Vec3,
    amount: number,
    index: number,
): Vec3 => {
    const shade = mix(1 - amount, 1 + amount, rand01(index, 11));

    return [
        clamp(base[0] * shade + mix(-amount, amount, rand01(index, 17)) * 0.08, 0, 1),
        clamp(base[1] * shade + mix(-amount, amount, rand01(index, 23)) * 0.08, 0, 1),
        clamp(base[2] * shade + mix(-amount, amount, rand01(index, 31)) * 0.08, 0, 1),
    ];
};

export const treeModelMeshMaterial = (name: string) => {
    const normalizedName = name.toLowerCase();

    return /leaf|leav/.test(normalizedName)
        ? ParticleAppearanceMaterial.Leaf
        : ParticleAppearanceMaterial.Bark;
};

export const calculateTreeModelBounds = (meshes: TreeModelMesh[]) => {
    const min: Vec3 = [Infinity, Infinity, Infinity];
    const max: Vec3 = [-Infinity, -Infinity, -Infinity];

    for (const mesh of meshes) {
        for (const vertex of mesh.vertices) {
            min[0] = Math.min(min[0], vertex[0]);
            min[1] = Math.min(min[1], vertex[1]);
            min[2] = Math.min(min[2], vertex[2]);
            max[0] = Math.max(max[0], vertex[0]);
            max[1] = Math.max(max[1], vertex[1]);
            max[2] = Math.max(max[2], vertex[2]);
        }
    }

    if (!min.every(Number.isFinite) || !max.every(Number.isFinite)) {
        throw new Error("tree model contains no finite vertices");
    }

    return { min, max };
};

const minVertexZ = (meshes: TreeModelMesh[]) => {
    let minZ = Infinity;

    for (const mesh of meshes) {
        for (const vertex of mesh.vertices) {
            minZ = Math.min(minZ, vertex[2]);
        }
    }

    return minZ;
};

export const fitTreeModelToEnvironment = (meshes: TreeModelMesh[]): TreeModelFit => {
    const { min, max } = calculateTreeModelBounds(meshes);
    const leafMeshes = meshes.filter((mesh) =>
        treeModelMeshMaterial(mesh.name) === ParticleAppearanceMaterial.Leaf
    );
    const sourceGroundZ = leafMeshes.length > 0
        ? minVertexZ(leafMeshes)
        : min[2];
    const sourceCenterXY = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
    ] as const;
    const targetCenterXY = [
        (DOMAIN_MIN[0] + DOMAIN_MAX[0]) * 0.5,
        (DOMAIN_MIN[1] + DOMAIN_MAX[1]) * 0.5,
    ] as const;
    const targetExtents: Vec3 = [
        DOMAIN_MAX[0] - DOMAIN_MIN[0] - DOMAIN_PADDING * 2,
        DOMAIN_MAX[1] - DOMAIN_MIN[1] - DOMAIN_PADDING * 2,
        DOMAIN_MAX[2] - DOMAIN_MIN[2] - DOMAIN_PADDING * 2,
    ];
    const sourceExtents: Vec3 = [
        max[0] - min[0],
        max[1] - min[1],
        max[2] - min[2],
    ];
    const sourceBelowGroundDepth = sourceGroundZ - min[2];
    const scale = Math.min(
        sourceExtents[0] > 1e-12 ? targetExtents[0] / sourceExtents[0] : Infinity,
        sourceExtents[1] > 1e-12 ? targetExtents[1] / sourceExtents[1] : Infinity,
        sourceExtents[2] > 1e-12
            ? targetExtents[2] / sourceExtents[2]
            : Infinity,
    );

    if (!Number.isFinite(scale) || scale <= 0) {
        throw new Error("tree model has no usable extent");
    }

    const groundBottomZ = DOMAIN_MIN[2] + DOMAIN_PADDING;
    const groundTopZ = groundBottomZ + sourceBelowGroundDepth * scale;

    return {
        meshes: meshes.map((mesh) => ({
            name: mesh.name,
            vertices: mesh.vertices.map((vertex): Vec3 => [
                (vertex[0] - sourceCenterXY[0]) * scale + targetCenterXY[0],
                (vertex[1] - sourceCenterXY[1]) * scale + targetCenterXY[1],
                (vertex[2] - sourceGroundZ) * scale + groundTopZ,
            ]),
        })),
        groundTopZ,
        groundBottomZ,
        scale,
    };
};

export const fitTreeModelMeshesToEnvironment = (meshes: TreeModelMesh[]) =>
    fitTreeModelToEnvironment(meshes).meshes;

const meshObjectFromVertices = (
    vertices: Vec3[],
    startVertex: number,
): SpawnMeshObject => {
    const min: Vec3 = [Infinity, Infinity, Infinity];
    const max: Vec3 = [-Infinity, -Infinity, -Infinity];

    for (const vertex of vertices) {
        min[0] = Math.min(min[0], vertex[0]);
        min[1] = Math.min(min[1], vertex[1]);
        min[2] = Math.min(min[2], vertex[2]);
        max[0] = Math.max(max[0], vertex[0]);
        max[1] = Math.max(max[1], vertex[1]);
        max[2] = Math.max(max[2], vertex[2]);
    }

    return {
        min,
        max,
        startVertex,
        countVertices: vertices.length,
    };
};

const flattenMeshes = (meshes: TreeModelMesh[]) => {
    const vertices: Vec3[] = [];
    const objects: SpawnMeshObject[] = [];

    for (const mesh of meshes) {
        const startVertex = vertices.length;
        for (const vertex of mesh.vertices) {
            vertices.push(vertex);
        }
        objects.push(meshObjectFromVertices(mesh.vertices, startVertex));
    }

    return { vertices, objects };
};

const writeMaterialAndClampPoints = (
    points: Float32Array,
    material: ParticleAppearanceMaterial,
) => {
    for (let i = 0; i < points.length; i += 4) {
        points[i] = clamp(points[i], DOMAIN_MIN[0] + SPAWN_CLAMP_PADDING, DOMAIN_MAX[0] - SPAWN_CLAMP_PADDING);
        points[i + 1] = clamp(points[i + 1], DOMAIN_MIN[1] + SPAWN_CLAMP_PADDING, DOMAIN_MAX[1] - SPAWN_CLAMP_PADDING);
        points[i + 2] = clamp(points[i + 2], DOMAIN_MIN[2] + SPAWN_CLAMP_PADDING, DOMAIN_MAX[2] - SPAWN_CLAMP_PADDING);
        points[i + 3] = material;
    }
};

const writePoint = (
    points: Float32Array,
    index: number,
    pos: Vec3,
    material: ParticleAppearanceMaterial,
) => {
    const offset = index * 4;

    points[offset] = clamp(pos[0], DOMAIN_MIN[0] + SPAWN_CLAMP_PADDING, DOMAIN_MAX[0] - SPAWN_CLAMP_PADDING);
    points[offset + 1] = clamp(pos[1], DOMAIN_MIN[1] + SPAWN_CLAMP_PADDING, DOMAIN_MAX[1] - SPAWN_CLAMP_PADDING);
    points[offset + 2] = clamp(pos[2], DOMAIN_MIN[2] + SPAWN_CLAMP_PADDING, DOMAIN_MAX[2] - SPAWN_CLAMP_PADDING);
    points[offset + 3] = material;
};

const buildBarkPoints = (
    meshes: TreeModelMesh[],
    nPoints: number,
) => {
    if (nPoints <= 0) {
        return new Float32Array();
    }

    const { vertices, objects } = flattenMeshes(meshes);
    const points = buildUniformSpawnPoints(vertices, {
        nPoints,
        objects,
    }).points;
    writeMaterialAndClampPoints(points, ParticleAppearanceMaterial.Bark);

    return points;
};

const buildLeafTriangles = (meshes: TreeModelMesh[]) => {
    const triangles: Triangle[] = [];
    let cumulativeArea = 0;

    for (const mesh of meshes) {
        for (let i = 0; i + 2 < mesh.vertices.length; i += 3) {
            const v0 = mesh.vertices[i];
            const v1 = mesh.vertices[i + 1];
            const v2 = mesh.vertices[i + 2];
            const area = triangleArea(v0, v1, v2);

            if (area <= 1e-10) {
                continue;
            }

            cumulativeArea += area;
            triangles.push({
                v0,
                v1,
                v2,
                normal: triangleNormal(v0, v1, v2),
                cumulativeArea,
            });
        }
    }

    if (triangles.length === 0) {
        throw new Error("tree model leaf meshes contain no usable triangle area");
    }

    return {
        triangles,
        totalArea: cumulativeArea,
    };
};

const triangleAtArea = (
    triangles: Triangle[],
    targetArea: number,
) => {
    let min = 0;
    let max = triangles.length - 1;

    while (min < max) {
        const mid = Math.floor((min + max) * 0.5);

        if (triangles[mid].cumulativeArea < targetArea) {
            min = mid + 1;
        } else {
            max = mid;
        }
    }

    return triangles[min];
};

const buildLeafPoints = (
    meshes: TreeModelMesh[],
    nPoints: number,
) => {
    if (nPoints <= 0) {
        return new Float32Array();
    }

    const { triangles, totalArea } = buildLeafTriangles(meshes);
    const points = new Float32Array(nPoints * 4);

    for (let i = 0; i < nPoints; i++) {
        const targetArea = ((i + rand01(i, 101)) / nPoints) * totalArea;
        const triangle = triangleAtArea(triangles, targetArea);
        const u = rand01(i, 103);
        const v = rand01(i, 107);
        const sqrtU = Math.sqrt(u);
        const w0 = 1 - sqrtU;
        const w1 = sqrtU * (1 - v);
        const w2 = sqrtU * v;
        // Leaves are authored as 2D sheets, so give them a sub-cell thickness
        // for MPM without pretending they are closed volumes.
        const normalOffset = (rand01(i, 109) - 0.5) * TREE_MODEL_LEAF_SURFACE_THICKNESS;
        const pos: Vec3 = [
            triangle.v0[0] * w0 + triangle.v1[0] * w1 + triangle.v2[0] * w2 + triangle.normal[0] * normalOffset,
            triangle.v0[1] * w0 + triangle.v1[1] * w1 + triangle.v2[1] * w2 + triangle.normal[1] * normalOffset,
            triangle.v0[2] * w0 + triangle.v1[2] * w1 + triangle.v2[2] * w2 + triangle.normal[2] * normalOffset,
        ];

        writePoint(points, i, pos, ParticleAppearanceMaterial.Leaf);
    }

    return points;
};

const writeAppearances = (
    appearances: Uint32Array,
    start: number,
    count: number,
    material: ParticleAppearanceMaterial,
) => {
    const baseColor = material === ParticleAppearanceMaterial.Leaf
        ? LEAF_BASE_COLOR
        : BARK_BASE_COLOR;
    const variation = material === ParticleAppearanceMaterial.Leaf
        ? 0.34
        : 0.28;

    for (let i = 0; i < count; i++) {
        appearances[start + i] = packParticleAppearance({
            color: colorVariation(baseColor, variation, start + i),
            material,
        });
    }
};

export const buildTreeModelSpawnPoints = ({
    meshes,
    nParticles,
    includeGround = true,
}: {
    meshes: TreeModelMesh[],
    nParticles: number,
    includeGround?: boolean,
}): TreeModelSpawnPoints => {
    if (!Number.isInteger(nParticles) || nParticles <= 0) {
        throw new Error("tree model requires a positive particle count");
    }

    const completeMeshes = meshes.filter((mesh) => mesh.vertices.length >= 3 && mesh.vertices.length % 3 === 0);
    if (completeMeshes.length === 0) {
        throw new Error("tree model contains no complete triangle meshes");
    }

    const fit = fitTreeModelToEnvironment(completeMeshes);
    const fittedMeshes = fit.meshes;
    const barkMeshes = fittedMeshes.filter((mesh) =>
        treeModelMeshMaterial(mesh.name) === ParticleAppearanceMaterial.Bark
    );
    const leafMeshes = fittedMeshes.filter((mesh) =>
        treeModelMeshMaterial(mesh.name) === ParticleAppearanceMaterial.Leaf
    );

    if (barkMeshes.length === 0 && leafMeshes.length === 0) {
        throw new Error("tree model contains no bark or leaf meshes");
    }

    const groundCount = includeGround && nParticles > 1
        ? Math.max(1, Math.floor(nParticles * GROUND_PARTICLE_FRACTION))
        : 0;
    const treeParticleCount = nParticles - groundCount;
    const ground = groundCount > 0
        ? buildProceduralGround({
            nParticles: groundCount,
            baseZ: fit.groundTopZ,
            thickness: fit.groundTopZ - fit.groundBottomZ,
        })
        : null;
    const leafCount = leafMeshes.length === 0
        ? 0
        : barkMeshes.length === 0
            ? treeParticleCount
            : Math.min(
                treeParticleCount - 1,
                Math.max(1, Math.round(treeParticleCount * LEAF_PARTICLE_FRACTION)),
            );
    const barkCount = treeParticleCount - leafCount;
    const barkPoints = buildBarkPoints(barkMeshes, barkCount);
    const leafPoints = buildLeafPoints(leafMeshes, leafCount);
    const spawnPoints = new Float32Array(nParticles * 4);
    const particleAppearances = new Uint32Array(nParticles);

    if (ground !== null) {
        spawnPoints.set(ground.spawnPoints);
        particleAppearances.set(ground.particleAppearances);
    }
    spawnPoints.set(barkPoints, groundCount * 4);
    spawnPoints.set(leafPoints, (groundCount + barkCount) * 4);
    writeAppearances(particleAppearances, groundCount, barkCount, ParticleAppearanceMaterial.Bark);
    writeAppearances(
        particleAppearances,
        groundCount + barkCount,
        leafCount,
        ParticleAppearanceMaterial.Leaf,
    );

    return {
        spawnPoints,
        particleAppearances,
        groundCount,
        barkCount,
        leafCount,
        groundTopZ: groundCount > 0 ? fit.groundTopZ : DEFAULT_GROUND_BASE_Z,
        groundBottomZ: groundCount > 0 ? fit.groundBottomZ : DEFAULT_GROUND_BASE_Z,
    };
};
