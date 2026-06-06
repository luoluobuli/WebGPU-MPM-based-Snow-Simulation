import {
    ParticleAppearanceMaterial,
    packParticleAppearance,
} from "$lib/gpu/particleAppearance/GpuParticleAppearanceBufferManager";

export type ProceduralForest = {
    spawnPoints: Float32Array;
    particleAppearances: Uint32Array;
};

type Vec3 = [number, number, number];

type TreeSpec = {
    base: Vec3;
    height: number;
    radius: number;
    canopyRadius: Vec3;
    canopyCenter: Vec3;
    conifer: boolean;
    lean: Vec3;
};

const DOMAIN_MIN = -5;
const DOMAIN_MAX = 5;
const GROUND_BASE_Z = -4.36;
const GROUND_THICKNESS = 0.96;

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

const terrainHeight = (x: number, y: number) =>
    GROUND_BASE_Z
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

const buildTreeSpecs = (seed: number): TreeSpec[] => {
    const bases: [number, number][] = [
        [-3.4, -2.6],
        [-1.4, -3.25],
        [1.0, -2.75],
        [3.2, -2.1],
        [-3.15, 0.2],
        [-0.75, -0.2],
        [1.85, 0.1],
        [-2.0, 2.6],
        [0.6, 2.25],
        [3.05, 2.5],
    ];

    return bases.map(([x, y], index) => {
        const jitterX = mix(-0.28, 0.28, hash01(seed, index, 1));
        const jitterY = mix(-0.28, 0.28, hash01(seed, index, 2));
        const baseX = x + jitterX;
        const baseY = y + jitterY;
        const baseZ = terrainHeight(baseX, baseY) + 0.1;
        const height = mix(3.3, 5.25, hash01(seed, index, 3));
        const radius = mix(0.16, 0.31, hash01(seed, index, 4));
        const canopyZ = baseZ + height * mix(0.68, 0.82, hash01(seed, index, 5));
        const conifer = hash01(seed, index, 6) > 0.58;
        const canopyRadius: Vec3 = conifer
            ? [
                mix(0.68, 0.96, hash01(seed, index, 7)),
                mix(0.68, 0.96, hash01(seed, index, 8)),
                mix(1.12, 1.52, hash01(seed, index, 9)),
            ]
            : [
                mix(0.85, 1.32, hash01(seed, index, 7)),
                mix(0.78, 1.22, hash01(seed, index, 8)),
                mix(0.58, 0.92, hash01(seed, index, 9)),
            ];

        return {
            base: [baseX, baseY, baseZ],
            height,
            radius,
            canopyRadius,
            canopyCenter: [
                baseX + mix(-0.24, 0.24, hash01(seed, index, 10)),
                baseY + mix(-0.24, 0.24, hash01(seed, index, 12)),
                Math.min(canopyZ, DOMAIN_MAX - canopyRadius[2] - 0.12),
            ],
            conifer,
            lean: [
                mix(-0.28, 0.28, hash01(seed, index, 13)),
                mix(-0.28, 0.28, hash01(seed, index, 14)),
                0,
            ],
        };
    });
};

const chooseTree = (trees: TreeSpec[], seed: number, index: number) =>
    trees[Math.floor(hash01(seed, index, 101) * trees.length) % trees.length];

const writeGroundParticle = (
    points: Float32Array,
    appearances: Uint32Array,
    index: number,
    seed: number,
) => {
    const x = mix(-4.75, 4.75, hash01(seed, index, 201));
    const y = mix(-4.75, 4.75, hash01(seed, index, 202));
    const top = terrainHeight(x, y);
    const z = top - GROUND_THICKNESS * hash01(seed, index, 203);
    const moss = hash01(seed, index, 204) > 0.74;
    const baseColor: Vec3 = moss ? [0.16, 0.23, 0.12] : [0.25, 0.17, 0.1];

    writePoint(points, index, [x, y, z], ParticleAppearanceMaterial.Soil);
    appearances[index] = packParticleAppearance({
        color: colorVariation(baseColor, 0.28, seed, index),
        material: ParticleAppearanceMaterial.Soil,
    });
};

const writeTrunkParticle = (
    points: Float32Array,
    appearances: Uint32Array,
    index: number,
    seed: number,
    trees: TreeSpec[],
) => {
    const tree = chooseTree(trees, seed, index);
    const t = hash01(seed, index, 301);
    const angle = hash01(seed, index, 302) * Math.PI * 2;
    const radial = Math.sqrt(hash01(seed, index, 303));
    const radius = tree.radius * mix(1, 0.42, t) * radial;
    const bendX = tree.lean[0] * t * t;
    const bendY = tree.lean[1] * t * t;
    const barkRidge = Math.sin(angle * 7 + t * 18) * 0.022;
    const x = tree.base[0] + bendX + Math.cos(angle) * (radius + barkRidge);
    const y = tree.base[1] + bendY + Math.sin(angle) * (radius + barkRidge);
    const z = tree.base[2] + tree.height * t;
    const barkBase: Vec3 = [0.31, 0.19, 0.1];

    writePoint(points, index, [x, y, z], ParticleAppearanceMaterial.Bark);
    appearances[index] = packParticleAppearance({
        color: colorVariation(barkBase, 0.34, seed, index),
        material: ParticleAppearanceMaterial.Bark,
    });
};

const branchPoint = (
    tree: TreeSpec,
    seed: number,
    index: number,
): Vec3 => {
    const branchIndex = Math.floor(hash01(seed, index, 401) * 7);
    const startT = mix(0.34, 0.82, hash01(seed, branchIndex, 402));
    const along = hash01(seed, index, 403);
    const azimuth = branchIndex * 2.39996 + mix(-0.35, 0.35, hash01(seed, branchIndex, 404));
    const length = tree.radius * mix(3.7, 6.8, hash01(seed, branchIndex, 405));
    const lift = mix(0.16, 0.48, hash01(seed, branchIndex, 406));
    const radius = tree.radius * mix(0.45, 0.18, along) * Math.sqrt(hash01(seed, index, 407));
    const radialAngle = hash01(seed, index, 408) * Math.PI * 2;
    const dirX = Math.cos(azimuth);
    const dirY = Math.sin(azimuth);
    const sideX = -dirY;
    const sideY = dirX;
    const start: Vec3 = [
        tree.base[0] + tree.lean[0] * startT * startT,
        tree.base[1] + tree.lean[1] * startT * startT,
        tree.base[2] + tree.height * startT,
    ];

    return [
        start[0] + dirX * length * along + sideX * Math.cos(radialAngle) * radius,
        start[1] + dirY * length * along + sideY * Math.cos(radialAngle) * radius,
        start[2] + lift * length * along + Math.sin(radialAngle) * radius,
    ];
};

const writeBranchParticle = (
    points: Float32Array,
    appearances: Uint32Array,
    index: number,
    seed: number,
    trees: TreeSpec[],
) => {
    const tree = chooseTree(trees, seed, index);
    const barkBase: Vec3 = [0.28, 0.17, 0.09];

    writePoint(points, index, branchPoint(tree, seed, index), ParticleAppearanceMaterial.Bark);
    appearances[index] = packParticleAppearance({
        color: colorVariation(barkBase, 0.32, seed, index),
        material: ParticleAppearanceMaterial.Bark,
    });
};

const writeLeafParticle = (
    points: Float32Array,
    appearances: Uint32Array,
    index: number,
    seed: number,
    trees: TreeSpec[],
) => {
    const tree = chooseTree(trees, seed, index);
    const theta = hash01(seed, index, 501) * Math.PI * 2;
    const u = hash01(seed, index, 502) * 2 - 1;
    const shell = Math.cbrt(hash01(seed, index, 503));
    const horizontal = Math.sqrt(Math.max(0, 1 - u * u));
    let localX = Math.cos(theta) * horizontal * shell;
    let localY = Math.sin(theta) * horizontal * shell;
    let localZ = u * shell;

    if (tree.conifer) {
        const v = hash01(seed, index, 504);
        localZ = mix(-1, 1, v);
        const coneRadius = (1 - Math.max(0, localZ) * 0.74) * Math.sqrt(hash01(seed, index, 505));
        localX = Math.cos(theta) * coneRadius;
        localY = Math.sin(theta) * coneRadius;
    }

    const x = tree.canopyCenter[0] + localX * tree.canopyRadius[0];
    const y = tree.canopyCenter[1] + localY * tree.canopyRadius[1];
    const z = tree.canopyCenter[2] + localZ * tree.canopyRadius[2];
    const baseColor: Vec3 = tree.conifer ? [0.1, 0.28, 0.15] : [0.18, 0.38, 0.15];

    writePoint(points, index, [x, y, z], ParticleAppearanceMaterial.Leaf);
    appearances[index] = packParticleAppearance({
        color: colorVariation(baseColor, 0.38, seed, index),
        material: ParticleAppearanceMaterial.Leaf,
    });
};

const writeRootParticle = (
    points: Float32Array,
    appearances: Uint32Array,
    index: number,
    seed: number,
    trees: TreeSpec[],
) => {
    const tree = chooseTree(trees, seed, index);
    const rootIndex = Math.floor(hash01(seed, index, 601) * 9);
    const angle = rootIndex * 2.39996 + mix(-0.24, 0.24, hash01(seed, rootIndex, 602));
    const along = hash01(seed, index, 603);
    const radius = tree.radius * mix(0.52, 0.16, along) * Math.sqrt(hash01(seed, index, 604));
    const radialAngle = hash01(seed, index, 605) * Math.PI * 2;
    const rootLength = tree.radius * mix(4.3, 7.4, hash01(seed, rootIndex, 606));
    const dirX = Math.cos(angle);
    const dirY = Math.sin(angle);
    const sideX = -dirY;
    const sideY = dirX;
    const sink = mix(0.02, 0.26, along);
    const x = tree.base[0] + dirX * rootLength * along + sideX * Math.cos(radialAngle) * radius;
    const y = tree.base[1] + dirY * rootLength * along + sideY * Math.cos(radialAngle) * radius;
    const z = terrainHeight(tree.base[0], tree.base[1]) - sink + Math.sin(radialAngle) * radius * 0.42;
    const barkBase: Vec3 = [0.24, 0.14, 0.07];

    writePoint(points, index, [x, y, z], ParticleAppearanceMaterial.Bark);
    appearances[index] = packParticleAppearance({
        color: colorVariation(barkBase, 0.3, seed, index),
        material: ParticleAppearanceMaterial.Bark,
    });
};

export const buildProceduralForest = ({
    nParticles,
    seed = 0x5650f017,
}: {
    nParticles: number,
    seed?: number,
}): ProceduralForest => {
    if (!Number.isInteger(nParticles) || nParticles <= 0) {
        throw new Error("procedural forest requires a positive particle count");
    }

    const points = new Float32Array(nParticles * 4);
    const appearances = new Uint32Array(nParticles);
    const trees = buildTreeSpecs(seed);
    const groundCount = Math.floor(nParticles * 0.34);
    const trunkCount = Math.floor(nParticles * 0.22);
    const rootCount = Math.floor(nParticles * 0.08);
    const branchCount = Math.floor(nParticles * 0.14);
    const rootStart = groundCount + trunkCount;
    const branchStart = rootStart + rootCount;
    const leafStart = branchStart + branchCount;

    for (let i = 0; i < groundCount; i++) {
        writeGroundParticle(points, appearances, i, seed);
    }

    for (let i = groundCount; i < groundCount + trunkCount; i++) {
        writeTrunkParticle(points, appearances, i, seed, trees);
    }

    for (let i = rootStart; i < branchStart; i++) {
        writeRootParticle(points, appearances, i, seed, trees);
    }

    for (let i = branchStart; i < leafStart; i++) {
        writeBranchParticle(points, appearances, i, seed, trees);
    }

    for (let i = leafStart; i < nParticles; i++) {
        writeLeafParticle(points, appearances, i, seed, trees);
    }

    return {
        spawnPoints: points,
        particleAppearances: appearances,
    };
};
