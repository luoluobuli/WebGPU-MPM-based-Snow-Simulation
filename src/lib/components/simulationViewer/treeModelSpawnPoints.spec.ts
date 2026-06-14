import { describe, expect, it } from "vitest";
import { ParticleAppearanceMaterial } from "$lib/gpu/particleAppearance/GpuParticleAppearanceBufferManager";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { collectTreeModelMeshes } from "./loadTreeModel";
import {
    buildTreeModelSpawnPoints,
    calculateTreeModelBounds,
    fitTreeModelToEnvironment,
    fitTreeModelMeshesToEnvironment,
    TREE_MODEL_LEAF_SURFACE_THICKNESS,
    treeModelMeshMaterial,
    type TreeModelMesh,
    type Vec3,
} from "./treeModelSpawnPoints";

const materialFromAppearance = (appearance: number) => appearance >>> 24;

const cubeVertices: Vec3[] = [
    [-1, -1, -1], [-1, 1, -1], [-1, 1, 1],
    [-1, -1, -1], [-1, 1, 1], [-1, -1, 1],
    [1, -1, -1], [1, 1, 1], [1, 1, -1],
    [1, -1, -1], [1, -1, 1], [1, 1, 1],
    [-1, -1, -1], [1, -1, 1], [1, -1, -1],
    [-1, -1, -1], [-1, -1, 1], [1, -1, 1],
    [-1, 1, -1], [1, 1, -1], [1, 1, 1],
    [-1, 1, -1], [1, 1, 1], [-1, 1, 1],
    [-1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, -1], [1, -1, -1], [1, 1, -1],
    [-1, -1, 1], [-1, 1, 1], [1, 1, 1],
    [-1, -1, 1], [1, 1, 1], [1, -1, 1],
];

const leafPlaneVertices: Vec3[] = [
    [-1.4, -1.4, 1.5], [1.4, -1.4, 1.5], [1.4, 1.4, 1.5],
    [-1.4, -1.4, 1.5], [1.4, 1.4, 1.5], [-1.4, 1.4, 1.5],
];

const materialCountsFromSpawnPoints = (points: Float32Array) => {
    const counts = new Map<number, number>();

    for (let i = 3; i < points.length; i += 4) {
        counts.set(points[i], (counts.get(points[i]) ?? 0) + 1);
    }

    return counts;
};

const expectPointsInsideDomain = (points: Float32Array) => {
    for (let i = 0; i < points.length; i += 4) {
        expect(Number.isFinite(points[i])).toBe(true);
        expect(Number.isFinite(points[i + 1])).toBe(true);
        expect(Number.isFinite(points[i + 2])).toBe(true);
        expect(points[i]).toBeGreaterThanOrEqual(-5);
        expect(points[i]).toBeLessThanOrEqual(5);
        expect(points[i + 1]).toBeGreaterThanOrEqual(-5);
        expect(points[i + 1]).toBeLessThanOrEqual(5);
        expect(points[i + 2]).toBeGreaterThanOrEqual(-5);
        expect(points[i + 2]).toBeLessThanOrEqual(5);
    }
};

const pointsForRange = (
    points: Float32Array,
    start: number,
    count: number,
) => {
    const result: Vec3[] = [];

    for (let i = start; i < start + count; i++) {
        const offset = i * 4;
        result.push([
            points[offset],
            points[offset + 1],
            points[offset + 2],
        ]);
    }

    return result;
};

const expectPointsInsideBounds = (
    points: Vec3[],
    {
        min,
        max,
    }: {
        min: Vec3,
        max: Vec3,
    },
    tolerance: number,
) => {
    for (const point of points) {
        expect(point[0]).toBeGreaterThanOrEqual(min[0] - tolerance);
        expect(point[0]).toBeLessThanOrEqual(max[0] + tolerance);
        expect(point[1]).toBeGreaterThanOrEqual(min[1] - tolerance);
        expect(point[1]).toBeLessThanOrEqual(max[1] + tolerance);
        expect(point[2]).toBeGreaterThanOrEqual(min[2] - tolerance);
        expect(point[2]).toBeLessThanOrEqual(max[2] + tolerance);
    }
};

const subtract = (a: Vec3, b: Vec3): Vec3 => [
    a[0] - b[0],
    a[1] - b[1],
    a[2] - b[2],
];

const dot = (a: Vec3, b: Vec3) =>
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

const distanceSquared = (a: Vec3, b: Vec3) => {
    const d = subtract(a, b);

    return dot(d, d);
};

const distanceSquaredToTriangle = (
    point: Vec3,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) => {
    const ab = subtract(b, a);
    const ac = subtract(c, a);
    const ap = subtract(point, a);
    const d1 = dot(ab, ap);
    const d2 = dot(ac, ap);

    if (d1 <= 0 && d2 <= 0) return distanceSquared(point, a);

    const bp = subtract(point, b);
    const d3 = dot(ab, bp);
    const d4 = dot(ac, bp);

    if (d3 >= 0 && d4 <= d3) return distanceSquared(point, b);

    const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        const v = d1 / (d1 - d3);
        const closest: Vec3 = [
            a[0] + ab[0] * v,
            a[1] + ab[1] * v,
            a[2] + ab[2] * v,
        ];

        return distanceSquared(point, closest);
    }

    const cp = subtract(point, c);
    const d5 = dot(ab, cp);
    const d6 = dot(ac, cp);

    if (d6 >= 0 && d5 <= d6) return distanceSquared(point, c);

    const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        const w = d2 / (d2 - d6);
        const closest: Vec3 = [
            a[0] + ac[0] * w,
            a[1] + ac[1] * w,
            a[2] + ac[2] * w,
        ];

        return distanceSquared(point, closest);
    }

    const va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
        const bc = subtract(c, b);
        const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const closest: Vec3 = [
            b[0] + bc[0] * w,
            b[1] + bc[1] * w,
            b[2] + bc[2] * w,
        ];

        return distanceSquared(point, closest);
    }

    const denom = 1 / (va + vb + vc);
    const v = vb * denom;
    const w = vc * denom;
    const closest: Vec3 = [
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w,
    ];

    return distanceSquared(point, closest);
};

const trianglesFromMeshes = (meshes: TreeModelMesh[]) => {
    const triangles: [Vec3, Vec3, Vec3][] = [];

    for (const mesh of meshes) {
        for (let i = 0; i + 2 < mesh.vertices.length; i += 3) {
            triangles.push([
                mesh.vertices[i],
                mesh.vertices[i + 1],
                mesh.vertices[i + 2],
            ]);
        }
    }

    return triangles;
};

const expectPointsNearMeshSurface = (
    points: Vec3[],
    meshes: TreeModelMesh[],
    tolerance: number,
) => {
    const triangles = trianglesFromMeshes(meshes);
    const maxSamples = 128;
    const stride = Math.max(1, Math.floor(points.length / maxSamples));

    for (let i = 0; i < points.length; i += stride) {
        const minDistanceSquared = triangles.reduce(
            (minDistance, triangle) =>
                Math.min(
                    minDistance,
                    distanceSquaredToTriangle(points[i], triangle[0], triangle[1], triangle[2]),
                ),
            Number.POSITIVE_INFINITY,
        );

        expect(Math.sqrt(minDistanceSquared)).toBeLessThanOrEqual(tolerance);
    }
};

describe("buildTreeModelSpawnPoints", () => {
    it("keeps named wood and 2D leaves as separate materials", () => {
        const nParticles = 128;
        const meshes: TreeModelMesh[] = [
            {
                name: "wood",
                vertices: cubeVertices,
            },
            {
                name: "leaves",
                vertices: leafPlaneVertices,
            },
        ];
        const tree = buildTreeModelSpawnPoints({
            meshes,
            nParticles,
            includeGround: false,
        });
        const spawnMaterials = materialCountsFromSpawnPoints(tree.spawnPoints);
        const appearanceMaterials = new Set(Array.from(tree.particleAppearances, materialFromAppearance));
        const leafPoints = pointsForRange(
            tree.spawnPoints,
            tree.barkCount,
            tree.leafCount,
        );
        const fittedLeafBounds = calculateTreeModelBounds(
            fitTreeModelMeshesToEnvironment(meshes)
                .filter((mesh) => treeModelMeshMaterial(mesh.name) === ParticleAppearanceMaterial.Leaf),
        );

        expect(tree.spawnPoints.length).toBe(nParticles * 4);
        expect(tree.particleAppearances.length).toBe(nParticles);
        expect(tree.groundCount).toBe(0);
        expect(tree.barkCount).toBeGreaterThan(0);
        expect(tree.leafCount).toBeGreaterThan(0);
        expect(tree.barkCount).toBeGreaterThan(tree.leafCount);
        expect(spawnMaterials.get(ParticleAppearanceMaterial.Bark)).toBe(tree.barkCount);
        expect(spawnMaterials.get(ParticleAppearanceMaterial.Leaf)).toBe(tree.leafCount);
        expect(appearanceMaterials.has(ParticleAppearanceMaterial.Bark)).toBe(true);
        expect(appearanceMaterials.has(ParticleAppearanceMaterial.Leaf)).toBe(true);
        expectPointsInsideBounds(leafPoints, fittedLeafBounds, 0.06);
        expectPointsNearMeshSurface(
            leafPoints,
            fitTreeModelMeshesToEnvironment(meshes)
                .filter((mesh) => treeModelMeshMaterial(mesh.name) === ParticleAppearanceMaterial.Leaf),
            TREE_MODEL_LEAF_SURFACE_THICKNESS * 0.5 + 0.01,
        );
        expectPointsInsideDomain(tree.spawnPoints);
    });

    it("surface-samples leaf-only 2D meshes with sub-cell thickness", () => {
        const nParticles = 64;
        const tree = buildTreeModelSpawnPoints({
            nParticles,
            includeGround: false,
            meshes: [
                {
                    name: "Leaf_876789",
                    vertices: leafPlaneVertices,
                },
            ],
        });
        const spawnMaterials = materialCountsFromSpawnPoints(tree.spawnPoints);
        const zRows = new Set<number>();

        for (let i = 2; i < tree.spawnPoints.length; i += 4) {
            zRows.add(Math.round(tree.spawnPoints[i] * 100_000));
        }

        expect(tree.groundCount).toBe(0);
        expect(tree.barkCount).toBe(0);
        expect(tree.leafCount).toBe(nParticles);
        expect(spawnMaterials.get(ParticleAppearanceMaterial.Leaf)).toBe(nParticles);
        expect(zRows.size).toBeGreaterThan(1);
        expectPointsInsideDomain(tree.spawnPoints);
    });

    it("loads the authored tree0.glb meshes with soil, bark, and visible leaf particles", async () => {
        const fsImport = "node:fs/promises";
        const { readFile } = await import(fsImport);
        const source = await readFile(new URL("../../assets/models/tree0.glb", import.meta.url)) as Uint8Array;
        const arrayBuffer = new ArrayBuffer(source.byteLength);
        new Uint8Array(arrayBuffer).set(source);
        const gltf = await new GLTFLoader().parseAsync(arrayBuffer, "");
        const meshes = collectTreeModelMeshes(gltf);
        const meshNames = meshes.map((mesh) => mesh.name);
        const tree = buildTreeModelSpawnPoints({
            meshes,
            nParticles: 2048,
        });
        const spawnMaterials = materialCountsFromSpawnPoints(tree.spawnPoints);
        const fit = fitTreeModelToEnvironment(meshes);
        const fittedMeshes = fit.meshes;
        const fittedLeafBounds = calculateTreeModelBounds(
            fittedMeshes.filter((mesh) =>
                treeModelMeshMaterial(mesh.name) === ParticleAppearanceMaterial.Leaf
            ),
        );
        const leafPoints = pointsForRange(
            tree.spawnPoints,
            tree.groundCount + tree.barkCount,
            tree.leafCount,
        );
        const originalWoodMesh = meshes.find((mesh) =>
            treeModelMeshMaterial(mesh.name) === ParticleAppearanceMaterial.Bark
        );
        const fittedWoodMesh = fittedMeshes.find((mesh) =>
            treeModelMeshMaterial(mesh.name) === ParticleAppearanceMaterial.Bark
        );
        const fittedRootVertices = originalWoodMesh && fittedWoodMesh
            ? fittedWoodMesh.vertices.filter((_, index) =>
                originalWoodMesh.vertices[index][2] < -0.02
            )
            : [];
        const minRootZ = Math.min(...fittedRootVertices.map((vertex) => vertex[2]));
        const maxRootZ = Math.max(...fittedRootVertices.map((vertex) => vertex[2]));
        const fittedLeafMeshes = fittedMeshes.filter((mesh) =>
            treeModelMeshMaterial(mesh.name) === ParticleAppearanceMaterial.Leaf
        );

        expect(meshNames.some((name) => name.includes("wood"))).toBe(true);
        expect(meshNames.some((name) => name.includes("leaves"))).toBe(true);
        expect(spawnMaterials.get(ParticleAppearanceMaterial.Soil)).toBe(tree.groundCount);
        expect(spawnMaterials.get(ParticleAppearanceMaterial.Bark)).toBe(tree.barkCount);
        expect(spawnMaterials.get(ParticleAppearanceMaterial.Leaf)).toBe(tree.leafCount);
        expect(tree.groundCount).toBeGreaterThan(2048 * 0.15);
        expect(tree.barkCount).toBeGreaterThan(2048 * 0.5);
        expect(tree.leafCount).toBeGreaterThan(2048 * 0.25);
        expect(tree.barkCount).toBeGreaterThan(tree.leafCount);
        expect(leafPoints.length).toBe(tree.leafCount);
        expectPointsInsideBounds(leafPoints, fittedLeafBounds, 0.06);
        expectPointsNearMeshSurface(
            leafPoints,
            fittedLeafMeshes,
            TREE_MODEL_LEAF_SURFACE_THICKNESS * 0.5 + 0.01,
        );
        expect(fittedRootVertices.length).toBeGreaterThan(0);
        expect(maxRootZ).toBeLessThanOrEqual(tree.groundTopZ + 1e-5);
        expect(minRootZ).toBeGreaterThanOrEqual(tree.groundBottomZ - 1e-5);
        expect(tree.groundTopZ - tree.groundBottomZ).toBeGreaterThan(3);
        expectPointsInsideDomain(tree.spawnPoints);
    });
});
