import type {
    AnimatedColliderGeometry,
    ColliderGeometry,
} from "../collider/GpuColliderBufferManager";

export type RigidPawSdfKinematicColliderAnimation = {
    primitiveCount: number,
    sdfResolution: number,
    packedPhi: Uint32Array,
    sampleWorldFromUnitAtTimeS: (
        timeS: number,
        target?: Float32Array<ArrayBuffer>,
    ) => Float32Array<ArrayBuffer>,
};

export type AnimatedRigidPawSdfKinematicColliderGeometry =
    AnimatedColliderGeometry
    & {
        kinematicCollider: RigidPawSdfKinematicColliderAnimation,
    };

export const isAnimatedRigidPawSdfKinematicColliderGeometry = (
    collider: ColliderGeometry | null | undefined,
): collider is AnimatedRigidPawSdfKinematicColliderGeometry =>
    collider !== null
    && collider !== undefined
    && "kinematicCollider" in collider
    && typeof collider.kinematicCollider === "object"
    && collider.kinematicCollider !== null
    && "packedPhi" in collider.kinematicCollider;
