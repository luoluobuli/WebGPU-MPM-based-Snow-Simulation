import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
    GpuKinematicColliderIntervalBatchBuffer,
} from "./GpuKinematicColliderIntervalBatchBuffer";

type BufferRecord = {
    descriptor: GPUBufferDescriptor,
    destroy: ReturnType<typeof vi.fn>,
    size: number,
};

const makeDevice = () => {
    const bufferRecords: BufferRecord[] = [];
    const uploads: Uint8Array[] = [];
    const createBuffer = vi.fn((descriptor: GPUBufferDescriptor) => {
        const record = {
            descriptor,
            destroy: vi.fn(),
            size: Number(descriptor.size),
        };
        bufferRecords.push(record);

        return record as unknown as GPUBuffer;
    });
    const writeBuffer = vi.fn((
        _buffer: GPUBuffer,
        _bufferOffset: GPUSize64,
        data: AllowSharedBufferSource,
        dataOffset = 0,
        size?: GPUSize64,
    ) => {
        const byteLength = Number(size);
        const source = ArrayBuffer.isView(data)
            ? new Uint8Array(
                data.buffer,
                data.byteOffset + Number(dataOffset),
                byteLength,
            )
            : new Uint8Array(data, Number(dataOffset), byteLength);
        uploads.push(new Uint8Array(source));
    });

    return {
        bufferRecords,
        device: {
            createBuffer,
            queue: { writeBuffer },
        } as unknown as GPUDevice,
        uploads,
        writeBuffer,
    };
};

describe("GpuKinematicColliderIntervalBatchBuffer", () => {
    beforeEach(() => {
        vi.stubGlobal("GPUBufferUsage", {
            COPY_DST: 0x0008,
            COPY_SRC: 0x0004,
        });
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it("packs every timed interval into one queue upload", () => {
        const {
            bufferRecords,
            device,
            uploads,
            writeBuffer,
        } = makeDevice();
        const batch = new GpuKinematicColliderIntervalBatchBuffer({
            device,
            intervalPoseByteLength: 8,
        });
        const intervals: Array<[number, number, number]> = [];

        batch.writeIntervals({
            intervalCount: 3,
            startTimeS: 2,
            simulationTimestepS: 0.25,
            writeInterval: ({
                intervalIndex,
                previousTimeS,
                currentTimeS,
                target,
            }) => {
                intervals.push([
                    intervalIndex,
                    previousTimeS,
                    currentTimeS,
                ]);
                target.fill(intervalIndex + 1);
            },
        });

        expect(bufferRecords[0].descriptor).toEqual({
            label: "kinematic collider interval pose batch buffer",
            size: 24,
            usage: 0x000c,
        });
        expect(intervals).toEqual([
            [0, 2, 2.25],
            [1, 2.25, 2.5],
            [2, 2.5, 2.75],
        ]);
        expect(writeBuffer).toHaveBeenCalledTimes(1);
        expect(Array.from(uploads[0])).toEqual([
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
        ]);
    });

    it("encodes ordered copies into the live pose buffer", () => {
        const { device } = makeDevice();
        const batch = new GpuKinematicColliderIntervalBatchBuffer({
            device,
            intervalPoseByteLength: 16,
        });
        batch.writeIntervals({
            intervalCount: 3,
            startTimeS: 0,
            simulationTimestepS: 0.1,
            writeInterval: () => {},
        });

        const copies: unknown[][] = [];
        const commandEncoder = {
            copyBufferToBuffer: (...args: unknown[]) => {
                copies.push(args);
            },
        } as unknown as GPUCommandEncoder;
        const targetBuffer = { size: 16 } as GPUBuffer;
        for (let intervalIndex = 0; intervalIndex < 3; intervalIndex++) {
            batch.copyIntervalTo({
                commandEncoder,
                intervalIndex,
                targetBuffer,
            });
        }

        expect(copies).toHaveLength(3);
        expect(copies.map((copy) => copy.slice(1))).toEqual([
            [0, targetBuffer, 0, 16],
            [16, targetBuffer, 0, 16],
            [32, targetBuffer, 0, 16],
        ]);
    });

    it("rejects invalid interval, timing, and copy ranges", () => {
        const { device } = makeDevice();
        expect(() =>
            new GpuKinematicColliderIntervalBatchBuffer({
                device,
                intervalPoseByteLength: 6,
            })
        ).toThrow("multiple of four");

        const batch = new GpuKinematicColliderIntervalBatchBuffer({
            device,
            intervalPoseByteLength: 16,
        });
        expect(() =>
            batch.writeIntervals({
                intervalCount: 0,
                startTimeS: 0,
                simulationTimestepS: 0.1,
                writeInterval: () => {},
            })
        ).toThrow("positive integer");
        expect(() =>
            batch.writeIntervals({
                intervalCount: 1,
                startTimeS: 0,
                simulationTimestepS: 0,
                writeInterval: () => {},
            })
        ).toThrow("positive timestep");
        expect(() =>
            batch.copyIntervalTo({
                commandEncoder: {} as GPUCommandEncoder,
                intervalIndex: 0,
                targetBuffer: { size: 16 } as GPUBuffer,
            })
        ).toThrow("has not been written");

        batch.writeIntervals({
            intervalCount: 1,
            startTimeS: 0,
            simulationTimestepS: 0.1,
            writeInterval: () => {},
        });
        expect(() =>
            batch.copyIntervalTo({
                commandEncoder: {} as GPUCommandEncoder,
                intervalIndex: 1,
                targetBuffer: { size: 16 } as GPUBuffer,
            })
        ).toThrow("has not been written");
        expect(() =>
            batch.copyIntervalTo({
                commandEncoder: {} as GPUCommandEncoder,
                intervalIndex: 0,
                targetBuffer: { size: 12 } as GPUBuffer,
            })
        ).toThrow("too small");
    });

    it("reuses capacity and retains growth buffers until destruction", () => {
        const { bufferRecords, device } = makeDevice();
        const batch = new GpuKinematicColliderIntervalBatchBuffer({
            device,
            intervalPoseByteLength: 4,
        });
        const writeIntervals = (intervalCount: number) => {
            batch.writeIntervals({
                intervalCount,
                startTimeS: 0,
                simulationTimestepS: 0.1,
                writeInterval: () => {},
            });
        };

        writeIntervals(1);
        writeIntervals(1);
        expect(bufferRecords).toHaveLength(1);

        writeIntervals(3);
        expect(bufferRecords).toHaveLength(2);
        expect(bufferRecords[0].destroy).not.toHaveBeenCalled();

        batch.destroy();

        expect(bufferRecords[0].destroy).toHaveBeenCalledOnce();
        expect(bufferRecords[1].destroy).toHaveBeenCalledOnce();
    });
});
