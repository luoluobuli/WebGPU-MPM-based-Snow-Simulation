const FNV_PRIME = 0x01000193;
const SECONDARY_PRIME = 0x5bd1e995;

type FingerprintState = {
    primary: number,
    secondary: number,
};

const createFingerprintState = (): FingerprintState => ({
    primary: 0x811c9dc5,
    secondary: 0x9747b28c,
});

const updateByte = (
    state: FingerprintState,
    value: number,
) => {
    state.primary = Math.imul(
        state.primary ^ value,
        FNV_PRIME,
    );
    state.secondary = Math.imul(
        state.secondary ^ value,
        SECONDARY_PRIME,
    );
};

const updateUint32 = (
    state: FingerprintState,
    value: number,
) => {
    updateByte(state, value & 0xff);
    updateByte(state, value >>> 8 & 0xff);
    updateByte(state, value >>> 16 & 0xff);
    updateByte(state, value >>> 24 & 0xff);
};

const updateBytes = (
    state: FingerprintState,
    values: Uint8Array,
) => {
    updateUint32(state, values.byteLength);
    for (const value of values) {
        updateByte(state, value);
    }
};

const digest = (state: FingerprintState) =>
    `${(state.primary >>> 0).toString(16).padStart(8, "0")}`
    + `${(state.secondary >>> 0).toString(16).padStart(8, "0")}`;

const bytesForBuffer = (
    buffer: ArrayBufferLike,
    byteOffset = 0,
    byteLength = buffer.byteLength - byteOffset,
) => new Uint8Array(
    buffer as ArrayBuffer,
    byteOffset,
    byteLength,
);

export const fingerprintBytes = (
    values: ArrayBufferLike | ArrayBufferView,
) => {
    const state = createFingerprintState();
    const bytes = ArrayBuffer.isView(values)
        ? bytesForBuffer(
            values.buffer,
            values.byteOffset,
            values.byteLength,
        )
        : bytesForBuffer(values);
    updateBytes(state, bytes);

    return digest(state);
};

export const fingerprintString = (value: string) =>
    fingerprintBytes(new TextEncoder().encode(value));

export const fingerprintFloat32Values = (
    values: ArrayLike<number>,
) => {
    const state = createFingerprintState();
    const scratch = new DataView(new ArrayBuffer(Float32Array.BYTES_PER_ELEMENT));
    updateUint32(state, values.length);

    for (let i = 0; i < values.length; i++) {
        scratch.setFloat32(0, values[i], true);
        for (let byte = 0; byte < Float32Array.BYTES_PER_ELEMENT; byte++) {
            updateByte(state, scratch.getUint8(byte));
        }
    }

    return digest(state);
};

export const combineCacheFingerprints = (
    fingerprints: readonly string[],
) => {
    const state = createFingerprintState();
    updateUint32(state, fingerprints.length);

    for (const fingerprint of fingerprints) {
        updateBytes(state, new TextEncoder().encode(fingerprint));
    }

    return digest(state);
};
