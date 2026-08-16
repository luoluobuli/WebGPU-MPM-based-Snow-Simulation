import {
    loadRigidPawSdfColliderAsset,
    type RigidPawSdfColliderAsset,
} from "$lib/gpu/collider/rigidPawSdfColliderAsset";
import type {
    AnimatedRigidPawSdfKinematicColliderGeometry,
} from "$lib/gpu/kinematicCollider/RigidPawSdfKinematicCollider";
import type {
    ColliderGeometry,
} from "$lib/gpu/collider/GpuColliderBufferManager";
import {
    AnimationMixer,
    Matrix3,
    Matrix4,
    Mesh,
    PropertyBinding,
    Quaternion,
    SkinnedMesh,
    Vector3,
    type Object3D,
} from "three";
import {
    GLTFLoader,
    type GLTF,
} from "three/addons/loaders/GLTFLoader.js";

import {
    ZUBEIA_PAW_COLLIDER_DEFINITIONS,
} from "./ZubeiaPawColliderCalibration";
import {
    applyZubeiaWalkRootPitchInPlace,
    zubeiaWalkBodySettleOffsetM,
    ZUBEIA_WALK_ROOT_PITCH_RAD,
} from "./ZubeiaWalkCalibration";
import {
    sampleZubeiaWalkTiming,
    ZUBEIA_WALK_ANIMATION_NAME,
} from "./ZubeiaWalkTiming";

const ANIMATION_DURATION_TOLERANCE_S = 1e-4;
const LEGACY_COLLIDER_MESH_PREFIX = "COLLIDER_";

type MeshRecord = {
    mesh: Mesh,
    vertexOffset: number,
    indexOffset: number,
    indexCount: number,
};

type DisplayMeshData = {
    records: MeshRecord[],
    indices: number[],
    uvs: number[],
    materialIndices: number[],
    vertexCount: number,
};

const traverseChildren = (
    object: Object3D,
    visit: (child: Object3D) => void,
) => {
    visit(object);

    for (const child of object.children) {
        traverseChildren(child, visit);
    }
};

const loadGltf = async (url: string | URL) =>
    await new Promise<GLTF>((resolve, reject) => {
        new GLTFLoader().load(
            url.toString(),
            resolve,
            undefined,
            reject,
        );
    });

export const findZubeiaWalkAnimation = (gltf: GLTF) => {
    const matches = gltf.animations.filter(
        clip => clip.name === ZUBEIA_WALK_ANIMATION_NAME,
    );

    if (matches.length !== 1) {
        throw new Error(
            `Zubeia GLB must contain exactly one `
            + `${ZUBEIA_WALK_ANIMATION_NAME} animation`,
        );
    }

    return matches[0];
};

const meshIsSkinned = (mesh: Mesh): mesh is SkinnedMesh =>
    mesh instanceof SkinnedMesh;

const collectDisplayMeshes = (gltf: GLTF): DisplayMeshData => {
    const records: MeshRecord[] = [];
    const indices: number[] = [];
    const uvs: number[] = [];
    const materialIndices: number[] = [];
    let vertexOffset = 0;

    traverseChildren(gltf.scene, child => {
        if (
            !(child instanceof Mesh)
            || child.name.startsWith(LEGACY_COLLIDER_MESH_PREFIX)
        ) {
            return;
        }

        const positionAttribute = child.geometry.attributes.position;
        if (positionAttribute === undefined) {
            return;
        }

        const uvAttribute = child.geometry.attributes.uv;
        const indexAttribute = child.geometry.index;
        const indexOffset = indices.length;
        const indexCount =
            indexAttribute?.count ?? positionAttribute.count;

        for (let index = 0; index < indexCount; index++) {
            indices.push(
                vertexOffset + (indexAttribute?.getX(index) ?? index),
            );
        }

        for (
            let vertexIndex = 0;
            vertexIndex < positionAttribute.count;
            vertexIndex++
        ) {
            if (uvAttribute === undefined) {
                uvs.push(0, 0);
            } else {
                uvs.push(
                    uvAttribute.getX(vertexIndex),
                    uvAttribute.getY(vertexIndex),
                );
            }
            materialIndices.push(0);
        }

        records.push({
            mesh: child,
            vertexOffset,
            indexOffset,
            indexCount,
        });
        vertexOffset += positionAttribute.count;
    });

    if (records.length === 0) {
        throw new Error("Zubeia GLB contains no display meshes");
    }
    if (!records.some(record => meshIsSkinned(record.mesh))) {
        throw new Error("Zubeia GLB contains no skinned display mesh");
    }

    return {
        records,
        indices,
        uvs,
        materialIndices,
        vertexCount: vertexOffset,
    };
};

const simPositionFromThree = (
    vector: Vector3,
): [number, number, number] => [
    vector.x,
    vector.z,
    vector.y,
];

const createGeometrySampler = ({
    gltf,
    mixer,
    asset,
    displayMeshes,
}: {
    gltf: GLTF,
    mixer: AnimationMixer,
    asset: RigidPawSdfColliderAsset,
    displayMeshes: DisplayMeshData,
}) => {
    const position = new Vector3();
    const sourceNormal = new Vector3();
    const normalEnd = new Vector3();
    const normal = new Vector3();
    const normalMatrix = new Matrix3();

    return (timeS: number): ColliderGeometry => {
        const {
            gaitTimeS,
            rootTranslation,
        } = sampleZubeiaWalkTiming({
            timeS,
            cycleDurationS: asset.cycleDurationS,
            durationS: asset.durationS,
            stridePerCycle: asset.stridePerCycle,
        });
        const bodySettleOffsetM = zubeiaWalkBodySettleOffsetM(timeS);
        const positions = new Array<number>(
            displayMeshes.vertexCount * 3,
        );
        const normals = new Array<number>(
            displayMeshes.vertexCount * 3,
        );
        const objects: ColliderGeometry["objects"] = [];

        mixer.setTime(gaitTimeS);
        gltf.scene.updateMatrixWorld(true);

        for (const record of displayMeshes.records) {
            const { mesh } = record;
            const positionAttribute =
                mesh.geometry.attributes.position;
            const normalAttribute = mesh.geometry.attributes.normal;
            const objectMin: [number, number, number] = [
                Infinity,
                Infinity,
                Infinity,
            ];
            const objectMax: [number, number, number] = [
                -Infinity,
                -Infinity,
                -Infinity,
            ];

            normalMatrix.getNormalMatrix(mesh.matrixWorld);
            if (meshIsSkinned(mesh)) {
                mesh.skeleton.update();
            }

            for (
                let vertexIndex = 0;
                vertexIndex < positionAttribute.count;
                vertexIndex++
            ) {
                position.fromBufferAttribute(
                    positionAttribute,
                    vertexIndex,
                );

                if (normalAttribute !== undefined) {
                    sourceNormal.fromBufferAttribute(
                        normalAttribute,
                        vertexIndex,
                    );
                    normalEnd.copy(position).add(sourceNormal);
                }

                if (meshIsSkinned(mesh)) {
                    mesh.applyBoneTransform(vertexIndex, position);

                    if (normalAttribute !== undefined) {
                        mesh.applyBoneTransform(
                            vertexIndex,
                            normalEnd,
                        );
                        normal.copy(normalEnd).sub(position);
                    }
                } else if (normalAttribute !== undefined) {
                    normal.copy(sourceNormal);
                }

                position.applyMatrix4(mesh.matrixWorld);
                const simPosition = simPositionFromThree(position);
                applyZubeiaWalkRootPitchInPlace(simPosition);
                simPosition[0] += rootTranslation[0];
                simPosition[1] += rootTranslation[1];
                simPosition[2] +=
                    rootTranslation[2] + bodySettleOffsetM;

                const outputOffset =
                    (record.vertexOffset + vertexIndex) * 3;
                positions[outputOffset] = simPosition[0];
                positions[outputOffset + 1] = simPosition[1];
                positions[outputOffset + 2] = simPosition[2];

                objectMin[0] = Math.min(
                    objectMin[0],
                    simPosition[0],
                );
                objectMin[1] = Math.min(
                    objectMin[1],
                    simPosition[1],
                );
                objectMin[2] = Math.min(
                    objectMin[2],
                    simPosition[2],
                );
                objectMax[0] = Math.max(
                    objectMax[0],
                    simPosition[0],
                );
                objectMax[1] = Math.max(
                    objectMax[1],
                    simPosition[1],
                );
                objectMax[2] = Math.max(
                    objectMax[2],
                    simPosition[2],
                );

                if (normalAttribute === undefined) {
                    normals[outputOffset] = 0;
                    normals[outputOffset + 1] = 0;
                    normals[outputOffset + 2] = 1;
                } else {
                    normal.applyMatrix3(normalMatrix).normalize();
                    const simNormal = simPositionFromThree(normal);
                    applyZubeiaWalkRootPitchInPlace(simNormal);
                    normals[outputOffset] = simNormal[0];
                    normals[outputOffset + 1] = simNormal[1];
                    normals[outputOffset + 2] = simNormal[2];
                }
            }

            objects.push({
                min: objectMin,
                max: objectMax,
                startIndex: record.indexOffset,
                countIndices: record.indexCount,
            });
        }

        return {
            positions,
            normals,
            uvs: displayMeshes.uvs,
            materialIndices: displayMeshes.materialIndices,
            textures: [],
            indices: displayMeshes.indices,
            objects,
        };
    };
};

const createSimFromThreeMatrix = () => {
    const cosine = Math.cos(ZUBEIA_WALK_ROOT_PITCH_RAD);
    const sine = Math.sin(ZUBEIA_WALK_ROOT_PITCH_RAD);

    return new Matrix4().set(
        1, 0, 0, 0,
        0, -sine, cosine, 0,
        0, cosine, sine, 0,
        0, 0, 0, 1,
    );
};

const findMotionBones = (gltf: GLTF) =>
    ZUBEIA_PAW_COLLIDER_DEFINITIONS.map(definition => {
        const sanitizedName = PropertyBinding.sanitizeNodeName(
            definition.motionBoneName,
        );
        const bone = gltf.scene.getObjectByName(sanitizedName);
        if (bone === undefined) {
            throw new Error(
                `Zubeia GLB has no motion bone `
                + `${definition.motionBoneName}`,
            );
        }

        return bone;
    });

const boundsMatrix = (
    volume: RigidPawSdfColliderAsset["volumes"][number],
) => {
    const center = new Vector3(
        (volume.minReference[0] + volume.maxReference[0]) * 0.5,
        (volume.minReference[1] + volume.maxReference[1]) * 0.5,
        (volume.minReference[2] + volume.maxReference[2]) * 0.5,
    );
    const halfExtent = new Vector3(
        (volume.maxReference[0] - volume.minReference[0]) * 0.5,
        (volume.maxReference[1] - volume.minReference[1]) * 0.5,
        (volume.maxReference[2] - volume.minReference[2]) * 0.5,
    );

    return new Matrix4().compose(
        center,
        new Quaternion(),
        halfExtent,
    );
};

const createRigidPawPoseSampler = ({
    gltf,
    mixer,
    asset,
}: {
    gltf: GLTF,
    mixer: AnimationMixer,
    asset: RigidPawSdfColliderAsset,
}) => {
    if (
        asset.volumeCount !== ZUBEIA_PAW_COLLIDER_DEFINITIONS.length
        || asset.volumes.some(
            (volume, index) =>
                volume.referenceFrameIndex
                !== ZUBEIA_PAW_COLLIDER_DEFINITIONS[
                    index
                ].referenceFrameIndex,
        )
    ) {
        throw new Error(
            "rigid paw SDF asset differs from paw calibration",
        );
    }

    const bones = findMotionBones(gltf);
    const simFromThree = createSimFromThreeMatrix();
    const threeFromSim = simFromThree.clone().invert();
    const referenceBoneWorldInverses =
        ZUBEIA_PAW_COLLIDER_DEFINITIONS.map(
            (definition, index) => {
                mixer.setTime(
                    definition.referenceFrameIndex
                    / asset.sampleRateHz,
                );
                gltf.scene.updateMatrixWorld(true);
                return bones[index].matrixWorld.clone().invert();
            },
        );
    const referenceBoundsMatrices =
        asset.volumes.map(boundsMatrix);
    const correctionM = new Float32Array(asset.volumeCount);
    const rootTranslationMatrix = new Matrix4();
    const relativeMatrix = new Matrix4();
    const worldMatrix = new Matrix4();

    return (
        timeS: number,
        target = new Float32Array(asset.volumeCount * 16),
    ) => {
        if (
            target.length !== asset.volumeCount * 16
            || !Number.isFinite(timeS)
        ) {
            throw new Error("rigid paw pose target or time is invalid");
        }

        const {
            gaitTimeS,
            rootTranslation,
        } = sampleZubeiaWalkTiming({
            timeS,
            cycleDurationS: asset.cycleDurationS,
            durationS: asset.durationS,
            stridePerCycle: asset.stridePerCycle,
        });
        const bodySettleOffsetM = zubeiaWalkBodySettleOffsetM(timeS);
        asset.sampleVerticalCorrectionAtGaitTimeS(
            gaitTimeS,
            correctionM,
        );
        mixer.setTime(gaitTimeS);
        gltf.scene.updateMatrixWorld(true);

        for (
            let pawIndex = 0;
            pawIndex < asset.volumeCount;
            pawIndex++
        ) {
            rootTranslationMatrix.makeTranslation(
                rootTranslation[0],
                rootTranslation[1],
                rootTranslation[2] + correctionM[pawIndex]
                    + bodySettleOffsetM,
            );
            relativeMatrix.copy(simFromThree)
                .multiply(bones[pawIndex].matrixWorld)
                .multiply(referenceBoneWorldInverses[pawIndex])
                .multiply(threeFromSim);
            worldMatrix.copy(rootTranslationMatrix)
                .multiply(relativeMatrix)
                .multiply(referenceBoundsMatrices[pawIndex]);
            target.set(worldMatrix.elements, pawIndex * 16);
        }

        return target;
    };
};

export const createZubeiaRigidPawWalkColliderGeometry = ({
    gltf,
    asset,
}: {
    gltf: GLTF,
    asset: RigidPawSdfColliderAsset,
}): AnimatedRigidPawSdfKinematicColliderGeometry => {
    const animation = findZubeiaWalkAnimation(gltf);
    if (
        Math.abs(animation.duration - asset.cycleDurationS)
            > ANIMATION_DURATION_TOLERANCE_S
    ) {
        throw new Error(
            "Zubeia display animation and rigid paw cycle differ",
        );
    }

    const mixer = new AnimationMixer(gltf.scene);
    mixer.clipAction(animation).play();
    const displayMeshes = collectDisplayMeshes(gltf);
    const sampleAtTimeS = createGeometrySampler({
        gltf,
        mixer,
        asset,
        displayMeshes,
    });
    const sampleWorldFromUnitAtTimeS =
        createRigidPawPoseSampler({
            gltf,
            mixer,
            asset,
        });
    const initialPose = sampleAtTimeS(0);

    return {
        ...initialPose,
        animationDurationS: asset.durationS,
        sampleAtTimeS,
        kinematicCollider: {
            primitiveCount: asset.volumeCount,
            sdfResolution: asset.sdfResolution,
            packedPhi: asset.packedPhi,
            sampleWorldFromUnitAtTimeS,
        },
    };
};

export const loadZubeiaRigidPawWalkCollider = async ({
    modelUrl,
    rigidPawSdfUrl,
}: {
    modelUrl: string | URL,
    rigidPawSdfUrl: string | URL,
}): Promise<AnimatedRigidPawSdfKinematicColliderGeometry> => {
    const [gltf, asset] = await Promise.all([
        loadGltf(modelUrl),
        loadRigidPawSdfColliderAsset(rigidPawSdfUrl),
    ]);

    return createZubeiaRigidPawWalkColliderGeometry({
        gltf,
        asset,
    });
};
