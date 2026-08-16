const RIGID_PAW_SDF_COLLIDER_MAGIC = 0x5344505a;
const RIGID_PAW_SDF_COLLIDER_VERSION = 1;
const HEADER_BYTE_LENGTH = 64;
const VOLUME_BYTE_LENGTH = 32;

export type RigidPawSdfVolume = {
    minReference: [number, number, number],
    maxReference: [number, number, number],
    referenceFrameIndex: number,
};

export type RigidPawSdfColliderAssetSource = {
    cycleCount: number,
    sdfResolution: number,
    sampleRateHz: number,
    cycleDurationS: number,
    durationS: number,
    stridePerCycle: [number, number, number],
    volumes: RigidPawSdfVolume[],
    verticalCorrectionM: Float32Array,
    packedPhi: Uint32Array,
};

export type RigidPawSdfColliderAsset = {
    volumeCount: number,
    motionFrameCount: number,
    cycleCount: number,
    sdfResolution: number,
    sampleRateHz: number,
    cycleDurationS: number,
    durationS: number,
    stridePerCycle: [number, number, number],
    volumes: readonly RigidPawSdfVolume[],
    verticalCorrectionM: Float32Array,
    packedPhi: Uint32Array,
    sampleVerticalCorrectionAtGaitTimeS: (
        gaitTimeS: number,
        target?: Float32Array<ArrayBuffer>,
    ) => Float32Array<ArrayBuffer>,
};

const assertPositiveInteger = (value: number, label: string) => {
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(`${label} must be a positive integer`);
    }
};

const assertFinite = (value: number, label: string) => {
    if (!Number.isFinite(value)) {
        throw new Error(`${label} must be finite`);
    }
};

const validateVolume = (
    volume: RigidPawSdfVolume,
    volumeIndex: number,
) => {
    assertPositiveInteger(
        volume.referenceFrameIndex + 1,
        `volume ${volumeIndex} reference frame`,
    );

    for (let axis = 0; axis < 3; axis++) {
        const minimum = volume.minReference[axis];
        const maximum = volume.maxReference[axis];
        assertFinite(
            minimum,
            `volume ${volumeIndex} minimum axis ${axis}`,
        );
        assertFinite(
            maximum,
            `volume ${volumeIndex} maximum axis ${axis}`,
        );
        if (maximum <= minimum) {
            throw new Error(
                `volume ${volumeIndex} bounds must have positive extent`,
            );
        }
    }
};

const motionFrameCountForSource = (
    source: RigidPawSdfColliderAssetSource,
) => Math.round(source.cycleDurationS * source.sampleRateHz) + 1;

const expectedPackedPhiWordCount = ({
    volumeCount,
    sdfResolution,
}: {
    volumeCount: number,
    sdfResolution: number,
}) => Math.ceil(volumeCount * sdfResolution ** 3 / 2);

const validateSource = (source: RigidPawSdfColliderAssetSource) => {
    assertPositiveInteger(source.cycleCount, "cycleCount");
    assertPositiveInteger(source.sdfResolution, "sdfResolution");
    if (source.sdfResolution < 4) {
        throw new Error("sdfResolution must be at least four");
    }
    if (source.volumes.length === 0) {
        throw new Error("volumes must not be empty");
    }

    assertFinite(source.sampleRateHz, "sampleRateHz");
    assertFinite(source.cycleDurationS, "cycleDurationS");
    assertFinite(source.durationS, "durationS");
    if (
        source.sampleRateHz <= 0
        || source.cycleDurationS <= 0
        || source.durationS <= 0
    ) {
        throw new Error("asset timing values must be positive");
    }
    if (
        Math.abs(
            source.durationS
            - source.cycleDurationS * source.cycleCount,
        ) > 1e-5
    ) {
        throw new Error(
            "durationS must equal cycleDurationS * cycleCount",
        );
    }

    source.stridePerCycle.forEach((value, axis) => {
        assertFinite(value, `stridePerCycle[${axis}]`);
    });
    source.volumes.forEach(validateVolume);

    const motionFrameCount = motionFrameCountForSource(source);
    const expectedCorrectionCount =
        motionFrameCount * source.volumes.length;
    if (
        source.verticalCorrectionM.length !== expectedCorrectionCount
        || source.verticalCorrectionM.some(value => !Number.isFinite(value))
    ) {
        throw new Error(
            "verticalCorrectionM does not match the motion dimensions",
        );
    }

    const expectedPhiWords = expectedPackedPhiWordCount({
        volumeCount: source.volumes.length,
        sdfResolution: source.sdfResolution,
    });
    if (source.packedPhi.length !== expectedPhiWords) {
        throw new Error(
            `packedPhi must contain ${expectedPhiWords} words`,
        );
    }
};

export const encodeRigidPawSdfColliderAsset = (
    source: RigidPawSdfColliderAssetSource,
) => {
    validateSource(source);

    const motionFrameCount = motionFrameCountForSource(source);
    const volumesOffset = HEADER_BYTE_LENGTH;
    const motionOffset =
        volumesOffset + source.volumes.length * VOLUME_BYTE_LENGTH;
    const fieldsOffset =
        motionOffset + source.verticalCorrectionM.byteLength;
    const byteLength = fieldsOffset + source.packedPhi.byteLength;
    const bytes = new Uint8Array(byteLength);
    const view = new DataView(bytes.buffer);

    view.setUint32(0, RIGID_PAW_SDF_COLLIDER_MAGIC, true);
    view.setUint16(4, RIGID_PAW_SDF_COLLIDER_VERSION, true);
    view.setUint16(6, HEADER_BYTE_LENGTH, true);
    view.setUint16(8, source.volumes.length, true);
    view.setUint16(10, source.sdfResolution, true);
    view.setUint16(12, motionFrameCount, true);
    view.setUint16(14, source.cycleCount, true);
    view.setFloat32(16, source.sampleRateHz, true);
    view.setFloat32(20, source.cycleDurationS, true);
    view.setFloat32(24, source.durationS, true);
    view.setFloat32(28, source.stridePerCycle[0], true);
    view.setFloat32(32, source.stridePerCycle[1], true);
    view.setFloat32(36, source.stridePerCycle[2], true);
    view.setUint32(40, volumesOffset, true);
    view.setUint32(44, motionOffset, true);
    view.setUint32(48, fieldsOffset, true);
    view.setUint32(52, byteLength, true);
    view.setUint32(56, source.packedPhi.length, true);
    view.setUint32(60, 0, true);

    source.volumes.forEach((volume, volumeIndex) => {
        const offset = volumesOffset + volumeIndex * VOLUME_BYTE_LENGTH;
        view.setUint32(offset, volumeIndex, true);
        view.setUint32(offset + 4, volume.referenceFrameIndex, true);

        for (let axis = 0; axis < 3; axis++) {
            view.setFloat32(
                offset + 8 + axis * Float32Array.BYTES_PER_ELEMENT,
                volume.minReference[axis],
                true,
            );
            view.setFloat32(
                offset + 20 + axis * Float32Array.BYTES_PER_ELEMENT,
                volume.maxReference[axis],
                true,
            );
        }
    });
    new Float32Array(
        bytes.buffer,
        motionOffset,
        source.verticalCorrectionM.length,
    ).set(source.verticalCorrectionM);
    new Uint32Array(
        bytes.buffer,
        fieldsOffset,
        source.packedPhi.length,
    ).set(source.packedPhi);

    return bytes;
};

const readVolume = ({
    view,
    volumesOffset,
    volumeIndex,
}: {
    view: DataView,
    volumesOffset: number,
    volumeIndex: number,
}): RigidPawSdfVolume => {
    const offset = volumesOffset + volumeIndex * VOLUME_BYTE_LENGTH;
    if (view.getUint32(offset, true) !== volumeIndex) {
        throw new Error(`volume ${volumeIndex} has an invalid index`);
    }

    const volume: RigidPawSdfVolume = {
        referenceFrameIndex: view.getUint32(offset + 4, true),
        minReference: [
            view.getFloat32(offset + 8, true),
            view.getFloat32(offset + 12, true),
            view.getFloat32(offset + 16, true),
        ],
        maxReference: [
            view.getFloat32(offset + 20, true),
            view.getFloat32(offset + 24, true),
            view.getFloat32(offset + 28, true),
        ],
    };
    validateVolume(volume, volumeIndex);

    return volume;
};

export const parseRigidPawSdfColliderAsset = (
    buffer: ArrayBuffer,
): RigidPawSdfColliderAsset => {
    if (buffer.byteLength < HEADER_BYTE_LENGTH) {
        throw new Error("rigid paw SDF asset is shorter than its header");
    }

    const view = new DataView(buffer);
    const magic = view.getUint32(0, true);
    const version = view.getUint16(4, true);
    const headerByteLength = view.getUint16(6, true);
    const volumeCount = view.getUint16(8, true);
    const sdfResolution = view.getUint16(10, true);
    const motionFrameCount = view.getUint16(12, true);
    const cycleCount = view.getUint16(14, true);
    const sampleRateHz = view.getFloat32(16, true);
    const cycleDurationS = view.getFloat32(20, true);
    const durationS = view.getFloat32(24, true);
    const stridePerCycle: [number, number, number] = [
        view.getFloat32(28, true),
        view.getFloat32(32, true),
        view.getFloat32(36, true),
    ];
    const volumesOffset = view.getUint32(40, true);
    const motionOffset = view.getUint32(44, true);
    const fieldsOffset = view.getUint32(48, true);
    const declaredByteLength = view.getUint32(52, true);
    const declaredPackedPhiWords = view.getUint32(56, true);

    if (magic !== RIGID_PAW_SDF_COLLIDER_MAGIC) {
        throw new Error("rigid paw SDF asset has an invalid magic value");
    }
    if (version !== RIGID_PAW_SDF_COLLIDER_VERSION) {
        throw new Error(
            `unsupported rigid paw SDF asset version ${version}`,
        );
    }
    if (headerByteLength !== HEADER_BYTE_LENGTH) {
        throw new Error("rigid paw SDF asset has an invalid header size");
    }

    assertPositiveInteger(volumeCount, "volumeCount");
    assertPositiveInteger(sdfResolution, "sdfResolution");
    assertPositiveInteger(motionFrameCount, "motionFrameCount");
    assertPositiveInteger(cycleCount, "cycleCount");
    if (sdfResolution < 4 || motionFrameCount < 2) {
        throw new Error("rigid paw SDF asset has invalid dimensions");
    }

    assertFinite(sampleRateHz, "sampleRateHz");
    assertFinite(cycleDurationS, "cycleDurationS");
    assertFinite(durationS, "durationS");
    stridePerCycle.forEach((value, axis) => {
        assertFinite(value, `stridePerCycle[${axis}]`);
    });
    if (
        sampleRateHz <= 0
        || cycleDurationS <= 0
        || durationS <= 0
        || Math.round(cycleDurationS * sampleRateHz) + 1
            !== motionFrameCount
        || Math.abs(durationS - cycleDurationS * cycleCount) > 1e-5
    ) {
        throw new Error("rigid paw SDF asset timing is inconsistent");
    }

    const expectedVolumesOffset = HEADER_BYTE_LENGTH;
    const expectedMotionOffset =
        expectedVolumesOffset + volumeCount * VOLUME_BYTE_LENGTH;
    const correctionValueCount = motionFrameCount * volumeCount;
    const expectedFieldsOffset =
        expectedMotionOffset
        + correctionValueCount * Float32Array.BYTES_PER_ELEMENT;
    const expectedPackedPhiWords = expectedPackedPhiWordCount({
        volumeCount,
        sdfResolution,
    });
    const expectedByteLength =
        expectedFieldsOffset
        + expectedPackedPhiWords * Uint32Array.BYTES_PER_ELEMENT;
    if (
        volumesOffset !== expectedVolumesOffset
        || motionOffset !== expectedMotionOffset
        || fieldsOffset !== expectedFieldsOffset
        || declaredPackedPhiWords !== expectedPackedPhiWords
        || declaredByteLength !== expectedByteLength
        || buffer.byteLength !== expectedByteLength
    ) {
        throw new Error(
            "rigid paw SDF asset sections are not tightly packed",
        );
    }

    const volumes = Array.from(
        { length: volumeCount },
        (_, volumeIndex) => readVolume({
            view,
            volumesOffset,
            volumeIndex,
        }),
    );
    volumes.forEach(volume => {
        if (volume.referenceFrameIndex >= motionFrameCount - 1) {
            throw new Error("volume reference frame is outside the cycle");
        }
    });

    const verticalCorrectionM = new Float32Array(
        buffer,
        motionOffset,
        correctionValueCount,
    );
    if (verticalCorrectionM.some(value => !Number.isFinite(value))) {
        throw new Error("rigid paw motion contains a non-finite value");
    }
    const packedPhi = new Uint32Array(
        buffer,
        fieldsOffset,
        expectedPackedPhiWords,
    );

    const sampleVerticalCorrectionAtGaitTimeS = (
        gaitTimeS: number,
        target = new Float32Array(volumeCount),
    ) => {
        assertFinite(gaitTimeS, "gaitTimeS");
        if (target.length !== volumeCount) {
            throw new Error(
                `correction target must contain ${volumeCount} values`,
            );
        }

        const clampedTimeS = Math.min(
            Math.max(gaitTimeS, 0),
            cycleDurationS,
        );
        const framePosition = clampedTimeS * sampleRateHz;
        const firstFrame = Math.min(
            Math.floor(framePosition),
            motionFrameCount - 1,
        );
        const secondFrame = Math.min(
            firstFrame + 1,
            motionFrameCount - 1,
        );
        const alpha = framePosition - firstFrame;
        const firstOffset = firstFrame * volumeCount;
        const secondOffset = secondFrame * volumeCount;

        for (
            let volumeIndex = 0;
            volumeIndex < volumeCount;
            volumeIndex++
        ) {
            const first = verticalCorrectionM[
                firstOffset + volumeIndex
            ];
            const second = verticalCorrectionM[
                secondOffset + volumeIndex
            ];
            target[volumeIndex] = first * (1 - alpha) + second * alpha;
        }

        return target;
    };

    return {
        volumeCount,
        motionFrameCount,
        cycleCount,
        sdfResolution,
        sampleRateHz,
        cycleDurationS,
        durationS,
        stridePerCycle,
        volumes,
        verticalCorrectionM,
        packedPhi,
        sampleVerticalCorrectionAtGaitTimeS,
    };
};

export const loadRigidPawSdfColliderAsset = async (
    url: string | URL,
) => {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(
            `failed to load rigid paw SDF asset: `
            + `${response.status} ${response.statusText}`,
        );
    }

    return parseRigidPawSdfColliderAsset(await response.arrayBuffer());
};
