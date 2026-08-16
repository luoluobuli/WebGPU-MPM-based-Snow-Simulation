import {
    AnimationClip,
    Bone,
    BufferGeometry,
    Float32BufferAttribute,
    Group,
    MeshBasicMaterial,
    Skeleton,
    SkinnedMesh,
    Uint16BufferAttribute,
} from "three";
import type { GLTF } from "three/addons/loaders/GLTFLoader.js";
import { describe, expect, it } from "vitest";

import {
    encodeRigidPawSdfColliderAsset,
    parseRigidPawSdfColliderAsset,
} from "$lib/gpu/collider/rigidPawSdfColliderAsset";

import {
    createZubeiaRigidPawWalkColliderGeometry,
    findZubeiaWalkAnimation,
} from "./loadZubeiaRigidPawWalkCollider";
import {
    ZUBEIA_PAW_COLLIDER_DEFINITIONS,
} from "./ZubeiaPawColliderCalibration";

import {
    ZUBEIA_WALK_BODY_SETTLE_M,
    ZUBEIA_WALK_ROOT_PITCH_RAD,
} from "./ZubeiaWalkCalibration";
const createTriangleGeometry = () => {
    const geometry = new BufferGeometry();

    geometry.setAttribute(
        "position",
        new Float32BufferAttribute([
            1, 2, 3,
            2, 2, 3,
            1, 3, 3,
        ], 3),
    );
    geometry.setAttribute(
        "normal",
        new Float32BufferAttribute([
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
        ], 3),
    );
    geometry.setAttribute(
        "uv",
        new Float32BufferAttribute([
            0, 0,
            1, 0,
            0, 1,
        ], 2),
    );
    geometry.setAttribute(
        "skinIndex",
        new Uint16BufferAttribute([
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        ], 4),
    );
    geometry.setAttribute(
        "skinWeight",
        new Float32BufferAttribute([
            1, 0, 0, 0,
            1, 0, 0, 0,
            1, 0, 0, 0,
        ], 4),
    );
    geometry.setIndex([0, 1, 2]);

    return geometry;
};

const createTestGltf = (animations = [
    new AnimationClip("walk_forward", 1.2, []),
]) => {
    const scene = new Group();
    const displayMesh = new SkinnedMesh(
        createTriangleGeometry(),
        new MeshBasicMaterial(),
    );
    const skinBone = new Bone();

    displayMesh.name = "body";
    displayMesh.add(skinBone);
    displayMesh.bind(new Skeleton([skinBone]));
    scene.add(displayMesh);

    ZUBEIA_PAW_COLLIDER_DEFINITIONS.forEach(definition => {
        const bone = new Bone();
        bone.name = definition.motionBoneName.replaceAll(".", "");
        scene.add(bone);
    });
    scene.updateMatrixWorld(true);

    return {
        animations,
        scene,
    } as unknown as GLTF;
};

const asset = () => parseRigidPawSdfColliderAsset(
    encodeRigidPawSdfColliderAsset({
        cycleCount: 3,
        sdfResolution: 4,
        sampleRateHz: 30,
        cycleDurationS: 1.2,
        durationS: 3.6,
        stridePerCycle: [0, 0.54, 0],
        volumes: ZUBEIA_PAW_COLLIDER_DEFINITIONS.map(
            definition => ({
                minReference: [-1, -1, -1],
                maxReference: [1, 1, 1],
                referenceFrameIndex:
                    definition.referenceFrameIndex,
            }),
        ),
        verticalCorrectionM: new Float32Array(37 * 4),
        packedPhi: new Uint32Array(128),
    }).buffer,
);

describe("loadZubeiaRigidPawWalkCollider", () => {
    it("requires the exact walk_forward animation", () => {
        expect(findZubeiaWalkAnimation(createTestGltf()).name).toBe(
            "walk_forward",
        );
        expect(() => findZubeiaWalkAnimation(createTestGltf([
            new AnimationClip("facial_idle", 1.2, []),
        ]))).toThrow("exactly one walk_forward");
    });

    it("uses matching display root motion and rigid paw transforms", () => {
        const collider = createZubeiaRigidPawWalkColliderGeometry({
            gltf: createTestGltf(),
            asset: asset(),
        });
        const middleDisplay = collider.sampleAtTimeS(1.8);
        const middlePaws =
            collider.kinematicCollider.sampleWorldFromUnitAtTimeS(
                1.8,
            );

        expect(collider.animationDurationS).toBeCloseTo(3.6);
        expect(collider.kinematicCollider.primitiveCount).toBe(4);
        expect(middleDisplay.positions).toHaveLength(9);
        expect(middleDisplay.indices).toEqual([0, 1, 2]);
        expect(middleDisplay.positions[1]).toBeCloseTo(
            3 * Math.cos(ZUBEIA_WALK_ROOT_PITCH_RAD)
            - 2 * Math.sin(ZUBEIA_WALK_ROOT_PITCH_RAD)
            + 0.81,
        );
        expect(middleDisplay.positions[2]).toBeCloseTo(
            3 * Math.sin(ZUBEIA_WALK_ROOT_PITCH_RAD)
            + 2 * Math.cos(ZUBEIA_WALK_ROOT_PITCH_RAD)
            - ZUBEIA_WALK_BODY_SETTLE_M,
        );
        expect(middlePaws[12]).toBeCloseTo(0);
        expect(middlePaws[13]).toBeCloseTo(0.81);
        expect(middlePaws[14]).toBeCloseTo(-ZUBEIA_WALK_BODY_SETTLE_M);
        expect(middlePaws[15]).toBeCloseTo(1);
    });

    it("rejects a display clip with different timing", () => {
        expect(() => createZubeiaRigidPawWalkColliderGeometry({
            gltf: createTestGltf([
                new AnimationClip("walk_forward", 2, []),
            ]),
            asset: asset(),
        })).toThrow("display animation and rigid paw cycle differ");
    });
});
