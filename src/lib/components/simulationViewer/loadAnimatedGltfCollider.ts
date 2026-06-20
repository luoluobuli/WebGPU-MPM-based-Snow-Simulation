import type { AnimatedColliderGeometry, ColliderGeometry } from "$lib/gpu/collider/GpuColliderBufferManager";
import {
    AnimationMixer,
    Matrix3,
    Mesh,
    PropertyBinding,
    SkinnedMesh,
    Vector3,
    type Object3D,
} from "three";
import { GLTFLoader, type GLTF } from "three/addons/loaders/GLTFLoader.js";

const KINEMATIC_LOOP_SECONDS = 1.2;
const KINEMATIC_FOOT_LIFT = 0.18;
const DEFAULT_KINEMATIC_SDF_RESOLUTION = 32;

const KINEMATIC_FOOT_NODE_NAMES = [
    "foot_ik.L",
    "foot_ik.R",
    "ORG-foot.L",
    "ORG-foot.R",
    "ORG-foot.L.001",
    "ORG-foot.R.001",
];

type MeshRecord = {
    mesh: Mesh,
    vertexOffset: number,
    indexOffset: number,
    indexCount: number,
};

type KinematicNode = {
    object: Object3D,
    baseY: number,
    phase: number,
};

const traverseChildren = (object: Object3D, fn: (child: Object3D) => void) => {
    fn(object);

    for (const child of object.children) {
        traverseChildren(child, fn);
    }
};

const loadGltf = async (url: string) =>
    await new Promise<GLTF>((resolve, reject) => {
        new GLTFLoader().load(url, resolve, undefined, reject);
    });

const positiveModulo = (value: number, period: number) =>
    ((value % period) + period) % period;

const simPositionFromThree = (vector: Vector3): [number, number, number] => [
    vector.x,
    vector.z,
    vector.y,
];


const meshIsSkinned = (mesh: Mesh): mesh is SkinnedMesh => mesh instanceof SkinnedMesh;

const findObjectByGltfSourceName = (gltf: GLTF, name: string) =>
    gltf.scene.getObjectByName(name)
    ?? gltf.scene.getObjectByName(PropertyBinding.sanitizeNodeName(name));

const collectKinematicNodes = (gltf: GLTF) => {
    const nodes: KinematicNode[] = [];

    for (const [index, name] of KINEMATIC_FOOT_NODE_NAMES.entries()) {
        const object = findObjectByGltfSourceName(gltf, name);
        if (object === undefined) continue;

        nodes.push({
            object,
            baseY: object.position.y,
            phase: index % 2 === 0 ? 0 : Math.PI,
        });
    }

    return nodes;
};

const applyKinematicFootMotion = (nodes: KinematicNode[], timeS: number) => {
    const loopTime = positiveModulo(timeS, KINEMATIC_LOOP_SECONDS);
    const angle = loopTime / KINEMATIC_LOOP_SECONDS * Math.PI * 2;

    for (const node of nodes) {
        const lift = Math.max(0, Math.sin(angle + node.phase)) * KINEMATIC_FOOT_LIFT;
        node.object.position.y = node.baseY + lift;
    }
};

const createAnimationMixer = (gltf: GLTF) => {
    if (gltf.animations.length === 0) return null;

    const mixer = new AnimationMixer(gltf.scene);
    mixer.clipAction(gltf.animations[0]).play();

    return mixer;
};

const collectMeshRecords = (gltf: GLTF) => {
    const records: MeshRecord[] = [];
    const indices: number[] = [];
    const uvs: number[] = [];
    const materialIndices: number[] = [];
    let vertexOffset = 0;

    traverseChildren(gltf.scene, child => {
        if (!(child instanceof Mesh)) return;

        const positionAttribute = child.geometry.attributes.position;
        if (positionAttribute === undefined) return;

        const uvAttribute = child.geometry.attributes.uv;
        const indexAttribute = child.geometry.index;
        const indexOffset = indices.length;
        const indexCount = indexAttribute?.count ?? positionAttribute.count;

        for (let i = 0; i < indexCount; i++) {
            indices.push(vertexOffset + (indexAttribute?.getX(i) ?? i));
        }

        for (let i = 0; i < positionAttribute.count; i++) {
            if (uvAttribute !== undefined) {
                uvs.push(uvAttribute.getX(i), uvAttribute.getY(i));
            } else {
                uvs.push(0, 0);
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

    return {
        records,
        indices,
        uvs,
        materialIndices,
        vertexCount: vertexOffset,
    };
};

const createGeometrySampler = ({
    gltf,
    records,
    indices,
    uvs,
    materialIndices,
    vertexCount,
    mixer,
    kinematicNodes,
    sdfResolution,
}: {
    gltf: GLTF,
    records: MeshRecord[],
    indices: number[],
    uvs: number[],
    materialIndices: number[],
    vertexCount: number,
    mixer: AnimationMixer | null,
    kinematicNodes: KinematicNode[],
    sdfResolution: number,
}) => {
    const position = new Vector3();
    const normal = new Vector3();
    const normalMatrix = new Matrix3();

    return (timeS: number): ColliderGeometry => {
        const positions: number[] = [];
        const normals: number[] = [];
        const objects: ColliderGeometry["objects"] = [];

        positions.length = vertexCount * 3;
        normals.length = vertexCount * 3;

        if (mixer !== null) {
            mixer.setTime(timeS);
        }
        applyKinematicFootMotion(kinematicNodes, timeS);
        gltf.scene.updateMatrixWorld(true);

        for (const record of records) {
            const { mesh } = record;
            const positionAttribute = mesh.geometry.attributes.position;
            const normalAttribute = mesh.geometry.attributes.normal;
            const objectMin: [number, number, number] = [Infinity, Infinity, Infinity];
            const objectMax: [number, number, number] = [-Infinity, -Infinity, -Infinity];
            normalMatrix.getNormalMatrix(mesh.matrixWorld);

            if (meshIsSkinned(mesh)) {
                mesh.skeleton.update();
            }

            for (let i = 0; i < positionAttribute.count; i++) {
                position.fromBufferAttribute(positionAttribute, i);
                if (meshIsSkinned(mesh)) {
                    mesh.applyBoneTransform(i, position);
                }
                position.applyMatrix4(mesh.matrixWorld);
                const simPosition = simPositionFromThree(position);
                const baseIndex = (record.vertexOffset + i) * 3;
                positions[baseIndex] = simPosition[0];
                positions[baseIndex + 1] = simPosition[1];
                positions[baseIndex + 2] = simPosition[2];

                objectMin[0] = Math.min(objectMin[0], simPosition[0]);
                objectMin[1] = Math.min(objectMin[1], simPosition[1]);
                objectMin[2] = Math.min(objectMin[2], simPosition[2]);
                objectMax[0] = Math.max(objectMax[0], simPosition[0]);
                objectMax[1] = Math.max(objectMax[1], simPosition[1]);
                objectMax[2] = Math.max(objectMax[2], simPosition[2]);

                if (normalAttribute !== undefined) {
                    normal.fromBufferAttribute(normalAttribute, i);
                    normal.applyMatrix3(normalMatrix).normalize();
                    const simNormal = simPositionFromThree(normal);
                    normals[baseIndex] = simNormal[0];
                    normals[baseIndex + 1] = simNormal[1];
                    normals[baseIndex + 2] = simNormal[2];
                } else {
                    normals[baseIndex] = 0;
                    normals[baseIndex + 1] = 0;
                    normals[baseIndex + 2] = 1;
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
            uvs,
            materialIndices,
            textures: [],
            indices,
            sdfResolution,
            objects,
        };
    };
};

export const loadAnimatedGltfCollider = async ({
    url,
    sdfResolution = DEFAULT_KINEMATIC_SDF_RESOLUTION,
}: {
    url: string,
    sdfResolution?: number,
}): Promise<AnimatedColliderGeometry> => {
    const gltf = await loadGltf(url);
    const mixer = createAnimationMixer(gltf);
    const kinematicNodes = collectKinematicNodes(gltf);
    const {
        records,
        indices,
        uvs,
        materialIndices,
        vertexCount,
    } = collectMeshRecords(gltf);

    if (!records.some(record => meshIsSkinned(record.mesh))) {
        throw new Error("kinematic GLB collider must contain at least one skinned mesh");
    }
    if (kinematicNodes.length === 0) {
        throw new Error("kinematic GLB collider did not expose known foot control nodes");
    }

    const sampleAtTimeS = createGeometrySampler({
        gltf,
        records,
        indices,
        uvs,
        materialIndices,
        vertexCount,
        mixer,
        kinematicNodes,
        sdfResolution,
    });
    const initialPose = sampleAtTimeS(0);
    const sourceDurationS = Math.max(
        ...gltf.animations.map(animation => animation.duration),
        KINEMATIC_LOOP_SECONDS,
    );

    return {
        ...initialPose,
        animationDurationS: sourceDurationS,
        sampleAtTimeS,
    };
};