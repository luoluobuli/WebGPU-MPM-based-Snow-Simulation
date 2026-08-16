import { mat4, type Mat4 } from "wgpu-matrix";
import type {
    RigidPawSdfKinematicColliderAnimation,
} from "./RigidPawSdfKinematicCollider";

export const RIGID_PAW_SDF_MATRIX_FLOATS = 16;
export const RIGID_PAW_SDF_POSE_FLOATS = 52;
export const RIGID_PAW_SDF_POSE_BYTES =
    RIGID_PAW_SDF_POSE_FLOATS * Float32Array.BYTES_PER_ELEMENT;

export const rigidPawSdfPoseBufferByteLength = (
    primitiveCount: number,
) => Math.max(
    RIGID_PAW_SDF_POSE_BYTES,
    primitiveCount * RIGID_PAW_SDF_POSE_BYTES,
);
const TIME_EPSILON_S = 1e-9;

const validateWorldFromUnit = ({
    matrices,
    primitiveCount,
    label,
}: {
    matrices: Float32Array,
    primitiveCount: number,
    label: string,
}) => {
    const expectedLength =
        primitiveCount * RIGID_PAW_SDF_MATRIX_FLOATS;
    if (
        matrices.length !== expectedLength
        || matrices.some(value => !Number.isFinite(value))
    ) {
        throw new Error(
            `${label} rigid paw pose must contain `
            + `${expectedLength} finite values`,
        );
    }
};

export const packRigidPawSdfColliderPose = ({
    currentWorldFromUnit,
    previousWorldFromUnit,
    timestepS,
    primitiveCount,
    target,
}: {
    currentWorldFromUnit: Float32Array,
    previousWorldFromUnit: Float32Array,
    timestepS: number,
    primitiveCount: number,
    target: Uint8Array,
}) => {
    validateWorldFromUnit({
        matrices: currentWorldFromUnit,
        primitiveCount,
        label: "current",
    });
    validateWorldFromUnit({
        matrices: previousWorldFromUnit,
        primitiveCount,
        label: "previous",
    });
    if (!Number.isFinite(timestepS) || timestepS <= 0) {
        throw new Error("rigid paw collider timestep must be positive");
    }

    const expectedByteLength =
        primitiveCount * RIGID_PAW_SDF_POSE_BYTES;
    if (
        target.byteLength !== expectedByteLength
        || target.byteOffset % Float32Array.BYTES_PER_ELEMENT !== 0
    ) {
        throw new Error(
            `rigid paw pose target must contain `
            + `${expectedByteLength} aligned bytes`,
        );
    }

    const values = new Float32Array(
        target.buffer,
        target.byteOffset,
        target.byteLength / Float32Array.BYTES_PER_ELEMENT,
    );

    for (
        let primitiveIndex = 0;
        primitiveIndex < primitiveCount;
        primitiveIndex++
    ) {
        const matrixOffset =
            primitiveIndex * RIGID_PAW_SDF_MATRIX_FLOATS;
        const poseOffset =
            primitiveIndex * RIGID_PAW_SDF_POSE_FLOATS;
        const current = currentWorldFromUnit.subarray(
            matrixOffset,
            matrixOffset + RIGID_PAW_SDF_MATRIX_FLOATS,
        ) as Mat4;
        const currentInverse = mat4.inverse(current);
        if (currentInverse.some(value => !Number.isFinite(value))) {
            throw new Error(
                `rigid paw ${primitiveIndex} transform is singular`,
            );
        }

        values.set(current, poseOffset);
        values.set(
            currentInverse,
            poseOffset + RIGID_PAW_SDF_MATRIX_FLOATS,
        );
        values.set(
            previousWorldFromUnit.subarray(
                matrixOffset,
                matrixOffset + RIGID_PAW_SDF_MATRIX_FLOATS,
            ),
            poseOffset + RIGID_PAW_SDF_MATRIX_FLOATS * 2,
        );
        values[poseOffset + 48] = timestepS;
        values[poseOffset + 49] = 0;
        values[poseOffset + 50] = 0;
        values[poseOffset + 51] = 0;
    }

    return target;
};

export class GpuRigidPawSdfKinematicColliderManager {
    readonly primitiveCount: number;
    readonly sdfResolution: number;
    readonly intervalPoseByteLength: number;
    readonly poseBuffer: GPUBuffer;
    readonly sdfBuffer: GPUBuffer;

    private readonly device: GPUDevice;
    private readonly animation:
        RigidPawSdfKinematicColliderAnimation | null;
    private readonly fallbackPoseUpload: Uint8Array;
    private readonly previousWorldFromUnit: Float32Array<ArrayBuffer>;
    private readonly currentWorldFromUnit: Float32Array<ArrayBuffer>;
    private readonly cachedWorldFromUnit: Float32Array<ArrayBuffer>;
    private cachedTimeS = Number.NaN;

    constructor({
        device,
        animation,
    }: {
        device: GPUDevice,
        animation:
            RigidPawSdfKinematicColliderAnimation | null,
    }) {
        const primitiveCount = animation?.primitiveCount ?? 0;
        const sdfResolution = animation?.sdfResolution ?? 2;
        if (
            !Number.isInteger(primitiveCount)
            || primitiveCount < 0
            || !Number.isInteger(sdfResolution)
            || sdfResolution < 2
        ) {
            throw new Error(
                "rigid paw SDF collider dimensions are invalid",
            );
        }

        const matrixValueCount =
            primitiveCount * RIGID_PAW_SDF_MATRIX_FLOATS;
        const intervalPoseByteLength =
            rigidPawSdfPoseBufferByteLength(primitiveCount);
        const fallbackPoseUpload =
            new Uint8Array(intervalPoseByteLength);
        const previousWorldFromUnit =
            new Float32Array(matrixValueCount);
        const currentWorldFromUnit =
            new Float32Array(matrixValueCount);
        const cachedWorldFromUnit =
            new Float32Array(matrixValueCount);
        const sdfUpload = animation?.packedPhi
            ?? new Uint32Array(1);
        const poseBuffer = device.createBuffer({
            label: primitiveCount > 0
                ? "rigid paw SDF pose buffer"
                : "disabled rigid paw SDF pose buffer",
            size: intervalPoseByteLength,
            usage: GPUBufferUsage.STORAGE
                | GPUBufferUsage.COPY_DST,
        });
        const sdfBuffer = device.createBuffer({
            label: primitiveCount > 0
                ? "static rigid paw SDF fields"
                : "disabled rigid paw SDF fields",
            size: Math.max(
                Uint32Array.BYTES_PER_ELEMENT,
                sdfUpload.byteLength,
            ),
            usage: GPUBufferUsage.STORAGE
                | GPUBufferUsage.COPY_DST,
        });

        this.device = device;
        this.animation = animation;
        this.primitiveCount = primitiveCount;
        this.sdfResolution = sdfResolution;
        this.intervalPoseByteLength = intervalPoseByteLength;
        this.poseBuffer = poseBuffer;
        this.sdfBuffer = sdfBuffer;
        this.fallbackPoseUpload = fallbackPoseUpload;
        this.previousWorldFromUnit = previousWorldFromUnit;
        this.currentWorldFromUnit = currentWorldFromUnit;
        this.cachedWorldFromUnit = cachedWorldFromUnit;

        device.queue.writeBuffer(
            poseBuffer,
            0,
            fallbackPoseUpload,
        );
        device.queue.writeBuffer(sdfBuffer, 0, sdfUpload);
    }

    packInterval({
        previousTimeS,
        currentTimeS,
        target,
    }: {
        previousTimeS: number,
        currentTimeS: number,
        target: Uint8Array,
    }) {
        if (this.animation === null) {
            target.fill(0);
            return;
        }
        if (
            !Number.isFinite(previousTimeS)
            || !Number.isFinite(currentTimeS)
            || currentTimeS <= previousTimeS
        ) {
            throw new Error(
                "rigid paw interval must have increasing finite times",
            );
        }

        if (
            Math.abs(this.cachedTimeS - previousTimeS)
                <= TIME_EPSILON_S
        ) {
            this.previousWorldFromUnit.set(
                this.cachedWorldFromUnit,
            );
        } else {
            this.animation.sampleWorldFromUnitAtTimeS(
                previousTimeS,
                this.previousWorldFromUnit,
            );
        }
        this.animation.sampleWorldFromUnitAtTimeS(
            currentTimeS,
            this.currentWorldFromUnit,
        );

        packRigidPawSdfColliderPose({
            currentWorldFromUnit: this.currentWorldFromUnit,
            previousWorldFromUnit: this.previousWorldFromUnit,
            timestepS: currentTimeS - previousTimeS,
            primitiveCount: this.primitiveCount,
            target,
        });
        this.cachedWorldFromUnit.set(this.currentWorldFromUnit);
        this.cachedTimeS = currentTimeS;
    }

    writeInterval(previousTimeS: number, currentTimeS: number) {
        if (this.animation === null) return;

        this.packInterval({
            previousTimeS,
            currentTimeS,
            target: this.fallbackPoseUpload,
        });
        this.device.queue.writeBuffer(
            this.poseBuffer,
            0,
            this.fallbackPoseUpload,
        );
    }

    destroy() {
        this.poseBuffer.destroy();
        this.sdfBuffer.destroy();
    }
}
