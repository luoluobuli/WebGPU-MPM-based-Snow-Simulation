import {
    ParticleAppearanceMaterial,
    packParticleAppearance,
} from "$lib/gpu/particleAppearance/GpuParticleAppearanceBufferManager";

type ProceduralGround = {
    spawnPoints: Float32Array,
    particleAppearances: Uint32Array,
};

type Vec3 = [number, number, number];

const DOMAIN_MIN = -5;
const DOMAIN_MAX = 5;
export const DEFAULT_GROUND_BASE_Z = -4.36;
const DEFAULT_GROUND_THICKNESS = 1.08;

const clamp = (value: number, min: number, max: number) =>
    Math.max(min, Math.min(max, value));

const mix = (a: number, b: number, t: number) => a + (b - a) * t;

const hashUint = (value: number) => {
    let x = value >>> 0;
    x ^= x >>> 16;
    x = Math.imul(x, 0x7feb352d);
    x ^= x >>> 15;
    x = Math.imul(x, 0x846ca68b);
    x ^= x >>> 16;

    return x >>> 0;
};

const hash01 = (seed: number, ...values: number[]) => {
    let hash = seed >>> 0;
    for (const value of values) {
        hash = hashUint(hash ^ Math.imul(value | 0, 0x9e3779b1));
    }

    return hash / 0xffffffff;
};

const terrainHeight = (
    x: number,
    y: number,
    baseZ = DEFAULT_GROUND_BASE_Z,
) =>
    baseZ
    + 0.1 * Math.sin(x * 1.4 + y * 0.35)
    + 0.07 * Math.cos(y * 1.15 - x * 0.28);

const writePoint = (
    points: Float32Array,
    index: number,
    pos: Vec3,
    material: ParticleAppearanceMaterial,
) => {
    const offset = index * 4;
    points[offset] = clamp(pos[0], DOMAIN_MIN + 0.08, DOMAIN_MAX - 0.08);
    points[offset + 1] = clamp(pos[1], DOMAIN_MIN + 0.08, DOMAIN_MAX - 0.08);
    points[offset + 2] = clamp(pos[2], DOMAIN_MIN + 0.08, DOMAIN_MAX - 0.08);
    points[offset + 3] = material;
};

const colorVariation = (
    base: Vec3,
    amount: number,
    seed: number,
    index: number,
): Vec3 => {
    const shade = mix(1 - amount, 1 + amount, hash01(seed, index, 11));

    return [
        clamp(base[0] * shade + mix(-amount, amount, hash01(seed, index, 17)) * 0.08, 0, 1),
        clamp(base[1] * shade + mix(-amount, amount, hash01(seed, index, 23)) * 0.08, 0, 1),
        clamp(base[2] * shade + mix(-amount, amount, hash01(seed, index, 31)) * 0.08, 0, 1),
    ];
};

const writeGroundParticle = (
    points: Float32Array,
    appearances: Uint32Array,
    index: number,
    seed: number,
    baseZ = DEFAULT_GROUND_BASE_Z,
    thickness = DEFAULT_GROUND_THICKNESS,
) => {
    const x = mix(-4.75, 4.75, hash01(seed, index, 201));
    const y = mix(-4.75, 4.75, hash01(seed, index, 202));
    const top = terrainHeight(x, y, baseZ);
    const z = top - thickness * hash01(seed, index, 203);
    const moss = hash01(seed, index, 204) > 0.74;
    const baseColor: Vec3 = moss ? [0.27, 0.38, 0.2] : [0.42, 0.31, 0.2];

    writePoint(points, index, [x, y, z], ParticleAppearanceMaterial.Soil);
    appearances[index] = packParticleAppearance({
        color: colorVariation(baseColor, 0.28, seed, index),
        material: ParticleAppearanceMaterial.Soil,
    });
};

export const buildProceduralGround = ({
    nParticles,
    seed = 0x5650f017,
    baseZ = DEFAULT_GROUND_BASE_Z,
    thickness = DEFAULT_GROUND_THICKNESS,
}: {
    nParticles: number,
    seed?: number,
    baseZ?: number,
    thickness?: number,
}): ProceduralGround => {
    if (!Number.isInteger(nParticles) || nParticles <= 0) {
        throw new Error("procedural ground requires a positive particle count");
    }

    const points = new Float32Array(nParticles * 4);
    const appearances = new Uint32Array(nParticles);

    for (let i = 0; i < nParticles; i++) {
        writeGroundParticle(points, appearances, i, seed, baseZ, thickness);
    }

    return {
        spawnPoints: points,
        particleAppearances: appearances,
    };
};
