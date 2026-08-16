import { describe, expect, it } from "vitest";

import matrixOpsSrc from "./matrixOps.wgsl?raw";
import plasticityOpsSrc from "./plasticityOps.wgsl?raw";

type Matrix3 = [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
];

type PlasticityCase = {
    name: string,
    deformation: Matrix3,
    expectedElastic: Matrix3,
};

const multiply = (left: Matrix3, right: Matrix3): Matrix3 => {
    const result = new Array<number>(9).fill(0);

    for (let column = 0; column < 3; column++) {
        for (let row = 0; row < 3; row++) {
            for (let inner = 0; inner < 3; inner++) {
                result[column * 3 + row] +=
                    left[inner * 3 + row]
                    * right[column * 3 + inner];
            }
        }
    }

    return result as Matrix3;
};

const transpose = (matrix: Matrix3): Matrix3 => {
    return [
        matrix[0], matrix[3], matrix[6],
        matrix[1], matrix[4], matrix[7],
        matrix[2], matrix[5], matrix[8],
    ];
};

const scaleColumns = (matrix: Matrix3, scale: [number, number, number]): Matrix3 => {
    return matrix.map((value, index) => value * scale[Math.floor(index / 3)]) as Matrix3;
};

const rotationX = (angle: number): Matrix3 => {
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);

    return [
        1, 0, 0,
        0, cosine, sine,
        0, -sine, cosine,
    ];
};

const rotationY = (angle: number): Matrix3 => {
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);

    return [
        cosine, 0, -sine,
        0, 1, 0,
        sine, 0, cosine,
    ];
};

const rotationZ = (angle: number): Matrix3 => {
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);

    return [
        cosine, sine, 0,
        -sine, cosine, 0,
        0, 0, 1,
    ];
};

const matrixFromSvd = (
    left: Matrix3,
    singulars: [number, number, number],
    right: Matrix3,
) => {
    return multiply(scaleColumns(left, singulars), transpose(right));
};

const determinant = (matrix: Matrix3) => {
    return matrix[0] * (matrix[4] * matrix[8] - matrix[7] * matrix[5])
        - matrix[3] * (matrix[1] * matrix[8] - matrix[7] * matrix[2])
        + matrix[6] * (matrix[1] * matrix[5] - matrix[4] * matrix[2]);
};

const reflectFirstColumn = (matrix: Matrix3): Matrix3 => {
    return matrix.map((value, index) => index < 3 ? -value : value) as Matrix3;
};

const expectMatrixClose = (
    actual: Matrix3,
    expected: Matrix3,
    tolerance = 8e-4,
) => {
    for (let i = 0; i < 9; i++) {
        expect(
            Math.abs(actual[i] - expected[i]),
            `matrix entry ${i}: expected ${expected[i]}, received ${actual[i]}`,
        ).toBeLessThanOrEqual(tolerance);
    }
};

const readMatrix = (values: Float32Array, offset: number): Matrix3 => {
    return [
        values[offset], values[offset + 1], values[offset + 2],
        values[offset + 4], values[offset + 5], values[offset + 6],
        values[offset + 8], values[offset + 9], values[offset + 10],
    ];
};

const makeCases = (): PlasticityCase[] => {
    const shearedLeft = multiply(rotationZ(0.57), rotationX(-0.31));
    const shearedRight = multiply(rotationY(0.42), rotationZ(-0.35));
    const reflectionLeft = reflectFirstColumn(
        multiply(rotationY(-0.28), rotationZ(0.41)),
    );
    const reflectionRight = multiply(rotationX(0.37), rotationY(0.19));
    const nearSingularLeft = multiply(rotationX(-0.44), rotationZ(0.22));
    const nearSingularRight = multiply(rotationZ(-0.33), rotationY(0.26));

    return [
        {
            name: "non-axis-aligned shear",
            deformation: matrixFromSvd(
                shearedLeft,
                [1.12, 0.91, 1.03],
                shearedRight,
            ),
            expectedElastic: matrixFromSvd(
                shearedLeft,
                [1.0075, 0.975, 1.0075],
                shearedRight,
            ),
        },
        {
            name: "reflected deformation",
            deformation: matrixFromSvd(
                reflectionLeft,
                [1.08, 0.94, 1.01],
                reflectionRight,
            ),
            expectedElastic: matrixFromSvd(
                reflectionLeft,
                [1.0075, 0.975, 1.0075],
                reflectionRight,
            ),
        },
        {
            name: "near-singular deformation",
            deformation: matrixFromSvd(
                nearSingularLeft,
                [1.1, 0.6, 1e-9],
                nearSingularRight,
            ),
            expectedElastic: matrixFromSvd(
                nearSingularLeft,
                [1.0075, 0.975, 0.975],
                nearSingularRight,
            ),
        },
    ];
};

const shaderSrc = `
${matrixOpsSrc}

const PARTICLE_MATERIAL_SOIL = 1u;
const PARTICLE_MATERIAL_BARK = 2u;
const PARTICLE_MATERIAL_LEAF = 3u;
const PARTICLE_MATERIAL_CRUST = 4u;

struct ParticleData {
    deformationElastic: mat3x3f,
    deformationPlastic: mat3x3f,
}

${plasticityOpsSrc}

struct PlasticityInput {
    columns: array<vec4f, 3>,
}

struct PlasticityOutput {
    elasticColumns: array<vec4f, 3>,
    plasticColumns: array<vec4f, 3>,
    metadata: vec4f,
}

@group(0) @binding(0) var<storage, read> inputs: array<PlasticityInput>;
@group(0) @binding(1) var<storage, read_write> outputs: array<PlasticityOutput>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
    let index = globalId.x;
    if index >= arrayLength(&inputs) {
        return;
    }

    var particle = ParticleData(
        mat3x3f(
            inputs[index].columns[0].xyz,
            inputs[index].columns[1].xyz,
            inputs[index].columns[2].xyz,
        ),
        IDENTITY_MAT3,
    );
    let yielded = applyPlasticity(&particle, 0u);

    outputs[index].elasticColumns[0] = vec4f(particle.deformationElastic[0], 0.0);
    outputs[index].elasticColumns[1] = vec4f(particle.deformationElastic[1], 0.0);
    outputs[index].elasticColumns[2] = vec4f(particle.deformationElastic[2], 0.0);
    outputs[index].plasticColumns[0] = vec4f(particle.deformationPlastic[0], 0.0);
    outputs[index].plasticColumns[1] = vec4f(particle.deformationPlastic[1], 0.0);
    outputs[index].plasticColumns[2] = vec4f(particle.deformationPlastic[2], 0.0);
    outputs[index].metadata = vec4f(select(0.0, 1.0, yielded), 0.0, 0.0, 0.0);
}
`;

describe("plasticity singular-value return mapping", () => {
    it("gives the walking-character crust a chunkier yield interval", () => {
        expect(plasticityOpsSrc).toContain("CRUST_CRITICAL_COMPRESSION = 6.0e-2");
        expect(plasticityOpsSrc).toContain("CRUST_CRITICAL_STRETCH = 2.0e-2");
        expect(plasticityOpsSrc).toContain("material == PARTICLE_MATERIAL_CRUST");
    });

    it("projects sheared, reflected, and near-singular deformation on the GPU", async () => {
        const adapter = await navigator.gpu?.requestAdapter();
        expect(adapter, "WebGPU adapter is required for the shader behavior test").not.toBeNull();
        if (adapter === null || adapter === undefined) {
            throw new Error("WebGPU adapter is required for the shader behavior test");
        }

        const device = await adapter.requestDevice();
        const cases = makeCases();
        const inputStrideFloats = 12;
        const outputStrideFloats = 28;
        const inputValues = new Float32Array(cases.length * inputStrideFloats);

        for (let caseIndex = 0; caseIndex < cases.length; caseIndex++) {
            const deformation = cases[caseIndex].deformation;
            const base = caseIndex * inputStrideFloats;

            for (let column = 0; column < 3; column++) {
                inputValues.set(
                    deformation.slice(column * 3, column * 3 + 3),
                    base + column * 4,
                );
            }
        }

        const shaderModule = device.createShaderModule({ code: shaderSrc });
        const compilationInfo = await shaderModule.getCompilationInfo();
        const compilationErrors = compilationInfo.messages
            .filter((message) => message.type === "error")
            .map((message) => `${message.lineNum}:${message.linePos} ${message.message}`);
        expect(compilationErrors).toEqual([]);

        const pipeline = await device.createComputePipelineAsync({
            layout: "auto",
            compute: {
                module: shaderModule,
                entryPoint: "main",
            },
        });
        const inputBuffer = device.createBuffer({
            size: inputValues.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        });
        const outputBuffer = device.createBuffer({
            size: cases.length * outputStrideFloats * Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
        });
        const readbackBuffer = device.createBuffer({
            size: outputBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        device.queue.writeBuffer(inputBuffer, 0, inputValues);
        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: inputBuffer } },
                { binding: 1, resource: { buffer: outputBuffer } },
            ],
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(cases.length);
        pass.end();
        encoder.copyBufferToBuffer(
            outputBuffer,
            0,
            readbackBuffer,
            0,
            outputBuffer.size,
        );
        device.queue.submit([encoder.finish()]);
        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const outputValues = new Float32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();

        for (let caseIndex = 0; caseIndex < cases.length; caseIndex++) {
            const testCase = cases[caseIndex];
            const base = caseIndex * outputStrideFloats;
            const elastic = readMatrix(outputValues, base);
            const plastic = readMatrix(outputValues, base + 12);
            const reconstructed = multiply(elastic, plastic);

            expect(outputValues[base + 24], `${testCase.name} must yield`).toBe(1);
            expect(elastic.every(Number.isFinite), `${testCase.name} elastic result`).toBe(true);
            expect(plastic.every(Number.isFinite), `${testCase.name} plastic result`).toBe(true);
            expectMatrixClose(elastic, testCase.expectedElastic);
            expectMatrixClose(reconstructed, testCase.deformation, 1.2e-3);
            expect(Math.sign(determinant(elastic))).toBe(
                Math.sign(determinant(testCase.deformation)),
            );
        }

        expect(
            Math.abs(determinant(readMatrix(outputValues, outputStrideFloats * 2))),
        ).toBeGreaterThan(0.9);

        inputBuffer.destroy();
        outputBuffer.destroy();
        readbackBuffer.destroy();
        device.destroy();
    });
});
