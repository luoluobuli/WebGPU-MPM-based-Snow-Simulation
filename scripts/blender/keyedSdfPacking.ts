const floatBits = new DataView(new ArrayBuffer(4));

export const float16BitsToFloat32 = (bits: number) => {
    const sign = (bits & 0x8000) << 16;
    const exponent = (bits >>> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    let floatExponent = 0;
    let floatMantissa = 0;

    if (exponent === 0) {
        if (mantissa === 0) {
            floatBits.setUint32(0, sign, false);
            return floatBits.getFloat32(0, false);
        }

        let normalizedMantissa = mantissa;
        let shift = 0;
        while ((normalizedMantissa & 0x400) === 0) {
            normalizedMantissa <<= 1;
            shift++;
        }
        normalizedMantissa &= 0x3ff;
        floatExponent = 127 - 15 - shift;
        floatMantissa = normalizedMantissa << 13;
    } else if (exponent === 0x1f) {
        floatExponent = 0xff;
        floatMantissa = mantissa << 13;
    } else {
        floatExponent = exponent - 15 + 127;
        floatMantissa = mantissa << 13;
    }

    floatBits.setUint32(
        0,
        sign | (floatExponent << 23) | floatMantissa,
        false,
    );
    return floatBits.getFloat32(0, false);
};
