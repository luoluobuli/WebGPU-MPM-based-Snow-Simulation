export const ZUBEIA_WALK_ROOT_PITCH_RAD =
    -4.074 * Math.PI / 180;
export const ZUBEIA_WALK_GROUND_SURFACE_Z_M = 0.07;
export const ZUBEIA_WALK_STRIDE_M = 0.54;

export const ZUBEIA_WALK_BODY_SETTLE_M = 0.075;
export const ZUBEIA_WALK_BODY_SETTLE_DURATION_S = 0.3;

// A kinematic collider has no mass response. This smooth prescribed settle
// supplies stance displacement without starting the ground inside a paw.
export const zubeiaWalkBodySettleOffsetM = (
    timeS: number,
) => {
    const progress = Math.min(
        1,
        Math.max(0, timeS / ZUBEIA_WALK_BODY_SETTLE_DURATION_S),
    );

    return -ZUBEIA_WALK_BODY_SETTLE_M * progress ** 2 * (3 - 2 * progress);
};

type MutableVectorValues = Float32Array | number[];

export const applyZubeiaWalkRootPitchInPlace = (
    values: MutableVectorValues,
    offset = 0,
) => {
    const y = values[offset + 1];
    const z = values[offset + 2];
    const cosine = Math.cos(ZUBEIA_WALK_ROOT_PITCH_RAD);
    const sine = Math.sin(ZUBEIA_WALK_ROOT_PITCH_RAD);

    values[offset + 1] = cosine * y - sine * z;
    values[offset + 2] = sine * y + cosine * z;

    return values;
};
