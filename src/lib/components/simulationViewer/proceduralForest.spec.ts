import { describe, expect, it } from "vitest";
import { ParticleAppearanceMaterial } from "$lib/gpu/particleAppearance/GpuParticleAppearanceBufferManager";
import { buildProceduralForest } from "./proceduralForest";

const materialFromPackedAppearance = (appearance: number) => appearance >>> 24;

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
            spawnMaterials.add(forest.spawnPoints[i]);
        }

        expect(materials.has(ParticleAppearanceMaterial.Soil)).toBe(true);
        expect(materials.has(ParticleAppearanceMaterial.Bark)).toBe(true);
        expect(materials.has(ParticleAppearanceMaterial.Leaf)).toBe(true);
        expect(spawnMaterials.has(ParticleAppearanceMaterial.Soil)).toBe(true);
        expect(spawnMaterials.has(ParticleAppearanceMaterial.Bark)).toBe(true);
        expect(spawnMaterials.has(ParticleAppearanceMaterial.Leaf)).toBe(true);
    });

    it("builds a deeper fuller ground layer", () => {
        const forest = buildProceduralForest({ nParticles: 2048, seed: 3456 });
        let soilCount = 0;
        let minSoilZ = Infinity;

        for (let i = 0; i < forest.particleAppearances.length; i++) {
            if (materialFromPackedAppearance(forest.particleAppearances[i]) !== ParticleAppearanceMaterial.Soil) {
                continue;
            }

            soilCount++;
            minSoilZ = Math.min(minSoilZ, forest.spawnPoints[i * 4 + 2]);
        }

        expect(soilCount).toBeGreaterThan(2048 * 0.3);
        expect(minSoilZ).toBeLessThan(-4.85);
    });
});
