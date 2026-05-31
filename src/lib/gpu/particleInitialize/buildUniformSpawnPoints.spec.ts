import { describe, expect, it } from "vitest";
import { buildUniformSpawnPoints } from "./buildUniformSpawnPoints";

const cubeVertices = [
    // x min
    [0, 0, 0], [0, 1, 0], [0, 1, 1],
    [0, 0, 0], [0, 1, 1], [0, 0, 1],
    // x max
    [1, 0, 0], [1, 1, 1], [1, 1, 0],
    [1, 0, 0], [1, 0, 1], [1, 1, 1],
    // y min
    [0, 0, 0], [1, 0, 1], [1, 0, 0],
    [0, 0, 0], [0, 0, 1], [1, 0, 1],
    // y max
    [0, 1, 0], [1, 1, 0], [1, 1, 1],
    [0, 1, 0], [1, 1, 1], [0, 1, 1],
    // z min
    [0, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 0], [1, 0, 0], [1, 1, 0],
    // z max
    [0, 0, 1], [0, 1, 1], [1, 1, 1],
    [0, 0, 1], [1, 1, 1], [1, 0, 1],
];

const translate = (
    vertices: number[][],
    offset: [number, number, number],
) => vertices.map((vertex) => [
    vertex[0] + offset[0],
    vertex[1] + offset[1],
    vertex[2] + offset[2],
]);

describe("buildUniformSpawnPoints", () => {
    it("builds exactly the requested number of points inside a cube", () => {
        const spawn = buildUniformSpawnPoints(cubeVertices, { nPoints: 128 });

        expect(spawn.pointCount).toBe(128);
        expect(spawn.points.length).toBe(128 * 4);
        expect(spawn.candidateCount).toBeGreaterThanOrEqual(128);

        for (let i = 0; i < spawn.points.length; i += 4) {
            expect(spawn.points[i]).toBeGreaterThan(0);
            expect(spawn.points[i]).toBeLessThan(1);
            expect(spawn.points[i + 1]).toBeGreaterThan(0);
            expect(spawn.points[i + 1]).toBeLessThan(1);
            expect(spawn.points[i + 2]).toBeGreaterThan(0);
            expect(spawn.points[i + 2]).toBeLessThan(1);
        }
    });

    it("does not bridge disconnected components", () => {
        const secondCube = translate(cubeVertices, [3, 0, 0]);
        const vertices = [...cubeVertices, ...secondCube];
        const spawn = buildUniformSpawnPoints(vertices, {
            nPoints: 128,
            objects: [
                { startVertex: 0, countVertices: cubeVertices.length },
                { startVertex: cubeVertices.length, countVertices: secondCube.length },
            ],
        });

        for (let i = 0; i < spawn.points.length; i += 4) {
            const x = spawn.points[i];
            expect(x < 1 || x > 3).toBe(true);
        }
    });

    it("is deterministic for the same input", () => {
        const first = buildUniformSpawnPoints(cubeVertices, { nPoints: 32 });
        const second = buildUniformSpawnPoints(cubeVertices, { nPoints: 32 });

        expect(Array.from(first.points)).toEqual(Array.from(second.points));
    });

    it("does not lock selected points to a few scanline rows", () => {
        const spawn = buildUniformSpawnPoints(cubeVertices, { nPoints: 128 });
        const roundedYRows = new Set<number>();

        for (let i = 1; i < spawn.points.length; i += 4) {
            roundedYRows.add(Math.round(spawn.points[i] * 10_000));
        }

        expect(roundedYRows.size).toBeGreaterThan(96);
    });
});
