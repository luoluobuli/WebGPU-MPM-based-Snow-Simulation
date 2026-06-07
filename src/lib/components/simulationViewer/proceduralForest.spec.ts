import { describe, expect, it } from "vitest";
import { ParticleAppearanceMaterial } from "$lib/gpu/particleAppearance/GpuParticleAppearanceBufferManager";
import { buildProceduralForest } from "./proceduralForest";

const materialFromPackedAppearance = (appearance: number) => appearance >>> 24;
const materialFromSpawnPoint = (material: number) => material;
const expectedTreeCenters: [number, number][] = [
    [-2.75, -2.2],
    [0.05, -2.55],
    [2.55, -1.75],
    [-2.45, 0.8],
    [0.55, 0.35],
    [2.35, 1.8],
];

type Point = {
    x: number,
    y: number,
    z: number,
};

const pointFromSpawnPoints = (spawnPoints: Float32Array, index: number): Point => {
    const offset = index * 4;

    return {
        x: spawnPoints[offset],
        y: spawnPoints[offset + 1],
        z: spawnPoints[offset + 2],
    };
};

const distanceSquared = (a: Point, b: Point) => {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = a.z - b.z;

    return dx * dx + dy * dy + dz * dz;
};

const horizontalDistanceSquared = (point: Point, center: [number, number]) => {
    const dx = point.x - center[0];
    const dy = point.y - center[1];

    return dx * dx + dy * dy;
};

describe("buildProceduralForest", () => {
    it("builds deterministic particle positions and appearances", () => {
        const first = buildProceduralForest({ nParticles: 512, seed: 1234 });
        const second = buildProceduralForest({ nParticles: 512, seed: 1234 });

        expect(Array.from(first.spawnPoints)).toEqual(Array.from(second.spawnPoints));
        expect(Array.from(first.particleAppearances)).toEqual(Array.from(second.particleAppearances));
    });

    it("builds exactly one finite domain-clamped point and appearance per particle", () => {
        const nParticles = 2048;
        const forest = buildProceduralForest({ nParticles, seed: 5678 });

        expect(forest.spawnPoints.length).toBe(nParticles * 4);
        expect(forest.particleAppearances.length).toBe(nParticles);

        for (let i = 0; i < forest.spawnPoints.length; i += 4) {
            expect(Number.isFinite(forest.spawnPoints[i])).toBe(true);
            expect(Number.isFinite(forest.spawnPoints[i + 1])).toBe(true);
            expect(Number.isFinite(forest.spawnPoints[i + 2])).toBe(true);
            expect(forest.spawnPoints[i]).toBeGreaterThanOrEqual(-5);
            expect(forest.spawnPoints[i]).toBeLessThanOrEqual(5);
            expect(forest.spawnPoints[i + 1]).toBeGreaterThanOrEqual(-5);
            expect(forest.spawnPoints[i + 1]).toBeLessThanOrEqual(5);
            expect(forest.spawnPoints[i + 2]).toBeGreaterThanOrEqual(-5);
            expect(forest.spawnPoints[i + 2]).toBeLessThanOrEqual(5);
        }
    });

    it("includes soil, bark, and leaf particles", () => {
        const forest = buildProceduralForest({ nParticles: 2048, seed: 9012 });
        const materials = new Set(Array.from(forest.particleAppearances, materialFromPackedAppearance));
        const spawnMaterials = new Set<number>();

        for (let i = 3; i < forest.spawnPoints.length; i += 4) {
            spawnMaterials.add(materialFromSpawnPoint(forest.spawnPoints[i]));
        }

        expect(materials.has(ParticleAppearanceMaterial.Soil)).toBe(true);
        expect(materials.has(ParticleAppearanceMaterial.Bark)).toBe(true);
        expect(materials.has(ParticleAppearanceMaterial.Leaf)).toBe(true);
        expect(spawnMaterials.has(ParticleAppearanceMaterial.Soil)).toBe(true);
        expect(spawnMaterials.has(ParticleAppearanceMaterial.Bark)).toBe(true);
        expect(spawnMaterials.has(ParticleAppearanceMaterial.Leaf)).toBe(true);
    });

    it("builds a deep ground layer with dense root support", () => {
        const forest = buildProceduralForest({ nParticles: 2048, seed: 3456 });
        let soilCount = 0;
        let barkCount = 0;
        let lowBarkCount = 0;
        let leafCount = 0;
        let minSoilZ = Infinity;

        for (let i = 0; i < forest.particleAppearances.length; i++) {
            const material = materialFromPackedAppearance(forest.particleAppearances[i]);
            const z = forest.spawnPoints[i * 4 + 2];

            if (material === ParticleAppearanceMaterial.Soil) {
                soilCount++;
                minSoilZ = Math.min(minSoilZ, z);
                continue;
            }

            if (material === ParticleAppearanceMaterial.Bark) {
                barkCount++;
                if (z < -4.25) {
                    lowBarkCount++;
                }
                continue;
            }

            if (material === ParticleAppearanceMaterial.Leaf) {
                leafCount++;
            }
        }

        expect(soilCount).toBeGreaterThan(2048 * 0.2);
        expect(barkCount).toBeGreaterThan(2048 * 0.55);
        expect(lowBarkCount).toBeGreaterThan(2048 * 0.08);
        expect(leafCount).toBeLessThan(2048 * 0.18);
        expect(minSoilZ).toBeLessThan(-4.85);
    });

    it("bridges every root system into its trunk base with bark crown particles", () => {
        const nParticles = 8192;
        const forest = buildProceduralForest({ nParticles, seed: 3456 });
        const groundCount = Math.floor(nParticles * 0.24);
        const trunkCount = Math.floor(nParticles * 0.3);
        const rootCount = Math.floor(nParticles * 0.13);
        const rootStart = groundCount + trunkCount;
        const rootEnd = rootStart + rootCount;
        const trunkParticles: Point[] = [];
        const rootParticles: Point[] = [];
        const treeNeighborhoodRadiusSquared = 0.76 * 0.76;
        const rootBridgeDistanceSquared = 0.28 * 0.28;

        for (let i = groundCount; i < rootStart; i++) {
            trunkParticles.push(pointFromSpawnPoints(forest.spawnPoints, i));
        }

        for (let i = rootStart; i < rootEnd; i++) {
            rootParticles.push(pointFromSpawnPoints(forest.spawnPoints, i));
        }

        for (const center of expectedTreeCenters) {
            const nearbyTrunkParticles = trunkParticles
                .filter(point => horizontalDistanceSquared(point, center) < treeNeighborhoodRadiusSquared)
                .sort((a, b) => a.z - b.z);
            const nearbyRootParticles = rootParticles
                .filter(point => horizontalDistanceSquared(point, center) < treeNeighborhoodRadiusSquared);
            const lowerTrunkParticles = nearbyTrunkParticles.slice(
                0,
                Math.ceil(nearbyTrunkParticles.length * 0.1),
            );
            const connectedLowerTrunkParticles = lowerTrunkParticles
                .filter(trunkPoint =>
                    nearbyRootParticles.some(rootPoint =>
                        distanceSquared(trunkPoint, rootPoint) < rootBridgeDistanceSquared,
                    ),
                );

            expect(nearbyTrunkParticles.length).toBeGreaterThan(240);
            expect(nearbyRootParticles.length).toBeGreaterThan(96);
            expect(connectedLowerTrunkParticles.length).toBeGreaterThan(
                lowerTrunkParticles.length * 0.68,
            );
        }
    });
});
