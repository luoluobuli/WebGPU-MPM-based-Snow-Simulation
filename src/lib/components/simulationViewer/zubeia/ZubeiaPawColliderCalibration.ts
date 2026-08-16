export const ZUBEIA_PAW_SDF_RESOLUTION = 80;
export const ZUBEIA_PAW_SDF_MAX_CELL_SIZE_M = 0.0125;

export type ZubeiaPawColliderDefinition = {
    name: "hindLeft" | "hindRight" | "frontLeft" | "frontRight",
    motionBoneName: string,
    referenceFrameIndex: number,
};

/**
 * Each field is the actual capped visible paw at its deepest stance frame.
 * The front fields follow the toe base because the wrist bone does not carry
 * their animated lift. The small baked z correction removes the remaining
 * non-rigid toe curl from the rigid transform without changing the field.
 */
export const ZUBEIA_PAW_COLLIDER_DEFINITIONS:
    readonly ZubeiaPawColliderDefinition[] = [
        {
            name: "hindLeft",
            motionBoneName: "DEF-foot.L.001",
            referenceFrameIndex: 0,
        },
        {
            name: "hindRight",
            motionBoneName: "DEF-foot.R.001",
            referenceFrameIndex: 18,
        },
        {
            name: "frontLeft",
            motionBoneName: "DEF-toe.L.001",
            referenceFrameIndex: 9,
        },
        {
            name: "frontRight",
            motionBoneName: "DEF-toe.R.001",
            referenceFrameIndex: 27,
        },
    ];
