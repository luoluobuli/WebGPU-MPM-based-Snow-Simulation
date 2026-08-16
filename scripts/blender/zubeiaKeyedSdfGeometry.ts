import {
    applyZubeiaWalkRootPitchInPlace,
} from "../../src/lib/components/simulationViewer/zubeia/ZubeiaWalkCalibration.ts";
import {
    AnimationMixer,
    Mesh,
    PropertyBinding,
    SkinnedMesh,
    Vector3,
    type Object3D,
} from "three";
import { type GLTF, GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import type { Vec3 } from "./keyedSdfBvh.ts";
import { createWeightedPawRegionIndices } from "./zubeiaPawRegion.ts";
import { growPawRegionsAboveContactBand } from "./zubeiaPawRegionGrow.ts";
import { closeWeightedPawRegions } from "./zubeiaPawClosure.ts";
import { collectPawRegionReferences } from "./zubeiaPawRegionReferences.ts";
import type {
    KeyedSdfManifest,
    MeshFrame,
    PawSurfaceReference,
    ZubeiaGeometrySource,
} from "./zubeiaKeyedSdfTypes.ts";

const COLLIDER_PREFIX = "COLLIDER_";

type MeshRecord = {
    mesh: SkinnedMesh,
    indices: Uint32Array,
};

const arrayBufferFromBytes = (bytes: Uint8Array) =>
    bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
    ) as ArrayBuffer;

const loadGltf = async (url: URL) => {
    const bytes = await Deno.readFile(url);

    return await new Promise<GLTF>((resolve, reject) => {
        new GLTFLoader().parse(
            arrayBufferFromBytes(bytes),
            "",
            resolve,
            reject,
        );
    });
};

const collectMeshRecords = (
    scene: Object3D,
    requestedMeshNames: readonly string[],
) => {
    const requested = new Set(requestedMeshNames);
    const records = new Map<string, MeshRecord>();

    scene.traverse(child => {
        if (
            !(child instanceof Mesh)
            || child.name.startsWith(COLLIDER_PREFIX)
            || !requested.has(child.name)
        ) {
            return;
        }
        if (!(child instanceof SkinnedMesh)) {
            throw new Error(`${child.name} must be a skinned mesh`);
        }
        if (records.has(child.name)) {
            throw new Error(`GLB has duplicate mesh name ${child.name}`);
        }

        const position = child.geometry.attributes.position;
        if (position === undefined) {
            throw new Error(`${child.name} has no position attribute`);
        }
        const sourceIndices = child.geometry.index;
        const indexCount = sourceIndices?.count ?? position.count;
        if (indexCount === 0 || indexCount % 3 !== 0) {
            throw new Error(
                `${child.name} must contain complete triangles`,
            );
        }

        const indices = new Uint32Array(indexCount);
        for (let index = 0; index < indexCount; index++) {
            indices[index] = sourceIndices?.getX(index) ?? index;
        }
        records.set(child.name, {
            mesh: child,
            indices,
        });
    });

    requested.forEach(meshName => {
        if (!records.has(meshName)) {
            throw new Error(`GLB has no solid mesh ${meshName}`);
        }
    });

    return records;
};

const updatePose = ({
    scene,
    mixer,
    records,
    timeS,
}: {
    scene: Object3D,
    mixer: AnimationMixer,
    records: ReadonlyMap<string, MeshRecord>,
    timeS: number,
}) => {
    mixer.setTime(timeS);
    scene.updateMatrixWorld(true);
    records.forEach(({ mesh }) => mesh.skeleton.update());
};

const sampleMeshPositions = (mesh: SkinnedMesh) => {
    const source = mesh.geometry.attributes.position;
    const positions = new Float32Array(source.count * 3);
    const point = new Vector3();

    for (let vertexIndex = 0; vertexIndex < source.count; vertexIndex++) {
        point.fromBufferAttribute(source, vertexIndex);
        mesh.applyBoneTransform(vertexIndex, point);
        point.applyMatrix4(mesh.matrixWorld);
        const offset = vertexIndex * 3;
        positions[offset] = point.x;
        positions[offset + 1] = point.z;
        positions[offset + 2] = point.y;
        applyZubeiaWalkRootPitchInPlace(positions, offset);
    }

    return positions;
};

const createFrameSampler = ({
    scene,
    mixer,
    records,
}: {
    scene: Object3D,
    mixer: AnimationMixer,
    records: ReadonlyMap<string, MeshRecord>,
}) =>
    (timeS: number) => {
        updatePose({
            scene,
            mixer,
            records,
            timeS,
        });
        const result = new Map<string, MeshFrame>();
        records.forEach(({ mesh, indices }, meshName) => {
            result.set(meshName, {
                positions: sampleMeshPositions(mesh),
                indices,
            });
        });

        return result;
    };

const descendantBoneNames = (
    mesh: SkinnedMesh,
    sourceRootName: string,
) => {
    const rootName = PropertyBinding.sanitizeNodeName(sourceRootName);
    const root = mesh.skeleton.bones.find(bone => bone.name === rootName);
    if (root === undefined) {
        throw new Error(
            `${mesh.name} has no deform root ${sourceRootName}`,
        );
    }

    const result = new Set<string>();
    root.traverse(child => result.add(child.name));
    return result;
};

const collectSurfaceReferences = ({
    records,
    meshNames,
    deformRoot,
    weightThreshold,
}: {
    records: ReadonlyMap<string, MeshRecord>,
    meshNames: readonly string[],
    deformRoot: string,
    weightThreshold: number,
}) => {
    const references: PawSurfaceReference[] = [];

    meshNames.forEach(meshName => {
        const { mesh } = records.get(meshName)!;
        const skinIndex = mesh.geometry.attributes.skinIndex;
        const skinWeight = mesh.geometry.attributes.skinWeight;
        if (skinIndex === undefined || skinWeight === undefined) {
            throw new Error(`${meshName} has no skin attributes`);
        }
        const descendants = descendantBoneNames(mesh, deformRoot);

        for (
            let vertexIndex = 0;
            vertexIndex < skinIndex.count;
            vertexIndex++
        ) {
            let descendantWeight = 0;
            for (let influence = 0; influence < 4; influence++) {
                const boneIndex = skinIndex.getComponent(
                    vertexIndex,
                    influence,
                );
                if (
                    descendants.has(mesh.skeleton.bones[boneIndex]?.name)
                ) {
                    descendantWeight += skinWeight.getComponent(
                        vertexIndex,
                        influence,
                    );
                }
            }
            if (descendantWeight >= weightThreshold) {
                references.push({
                    meshName,
                    vertexIndex,
                });
            }
        }
    });

    if (references.length < 32) {
        throw new Error(
            `${deformRoot} yielded only ${references.length} surface vertices`,
        );
    }

    return references;
};

const positionFromFrame = (
    frame: ReadonlyMap<string, MeshFrame>,
    reference: PawSurfaceReference,
): Vec3 => {
    const positions = frame.get(reference.meshName)!.positions;
    const offset = reference.vertexIndex * 3;

    return [
        positions[offset],
        positions[offset + 1],
        positions[offset + 2],
    ];
};

const measurePawBounds = ({
    frames,
    references,
    paddingM,
}: {
    frames: readonly ReadonlyMap<string, MeshFrame>[],
    references: readonly PawSurfaceReference[],
    paddingM: number,
}) => {
    const minLocal: Vec3 = [Infinity, Infinity, Infinity];
    const maxLocal: Vec3 = [-Infinity, -Infinity, -Infinity];

    frames.forEach(frame => {
        references.forEach(reference => {
            const position = positionFromFrame(frame, reference);
            for (let axis = 0; axis < 3; axis++) {
                minLocal[axis] = Math.min(
                    minLocal[axis],
                    position[axis] - paddingM,
                );
                maxLocal[axis] = Math.max(
                    maxLocal[axis],
                    position[axis] + paddingM,
                );
            }
        });
    });

    return {
        minLocal,
        maxLocal,
    };
};

export const validateKeyedSdfManifest = (
    manifest: KeyedSdfManifest,
) => {
    if (
        !Number.isInteger(manifest.action.fps)
        || manifest.action.fps <= 0
        || !Number.isInteger(manifest.action.cycle_intervals)
        || manifest.action.cycle_intervals <= 0
        || !Number.isInteger(manifest.action.preview_cycles)
        || manifest.action.preview_cycles <= 0
        || manifest.action.root_motion !== "in_place"
    ) {
        throw new Error("keyed SDF manifest has invalid action timing");
    }
    if (
        manifest.axes.forward.join(",") !== "0,-1,0"
        || manifest.axes.up.join(",") !== "0,0,1"
    ) {
        throw new Error("keyed SDF manifest has unsupported axes");
    }
    if (
        !Number.isFinite(manifest.sdf.bounds_padding_m)
        || manifest.sdf.bounds_padding_m <= 0
        || !Number.isFinite(manifest.sdf.topology_weld_tolerance_m)
        || manifest.sdf.topology_weld_tolerance_m <= 0
        || !Number.isFinite(manifest.sdf.minimum_closure_z_m)
        || manifest.sdf.minimum_closure_z_m <= 0
        || !Number.isFinite(manifest.sdf.surface_weight_threshold)
        || manifest.sdf.surface_weight_threshold <= 0
        || manifest.sdf.surface_weight_threshold > 1
    ) {
        throw new Error("keyed SDF manifest has invalid field settings");
    }
    if (manifest.paws.length !== 4) {
        throw new Error("keyed SDF manifest must define four paws");
    }

    const pawNames = new Set<string>();
    manifest.paws.forEach(paw => {
        if (
            pawNames.has(paw.name)
            || paw.solid_meshes.length === 0
            || new Set(paw.solid_meshes).size !== paw.solid_meshes.length
        ) {
            throw new Error(`keyed SDF paw ${paw.name} is invalid`);
        }
        pawNames.add(paw.name);
    });
};

export const loadZubeiaKeyedSdfGeometry = async ({
    sourceUrl,
    manifest,
}: {
    sourceUrl: URL,
    manifest: KeyedSdfManifest,
}): Promise<ZubeiaGeometrySource> => {
    validateKeyedSdfManifest(manifest);
    const gltf = await loadGltf(sourceUrl);
    const animationMatches = gltf.animations.filter(
        animation => animation.name === manifest.action.name,
    );
    if (animationMatches.length !== 1) {
        throw new Error(
            `GLB must contain one ${manifest.action.name} animation`,
        );
    }

    const animation = animationMatches[0];
    const cycleDurationS = manifest.action.cycle_intervals
        / manifest.action.fps;
    if (Math.abs(animation.duration - cycleDurationS) > 1e-5) {
        throw new Error(
            "GLB animation duration does not match keyed SDF manifest",
        );
    }

    const meshNames = [...new Set(
        manifest.paws.flatMap(paw => paw.solid_meshes),
    )];
    const records = collectMeshRecords(gltf.scene, meshNames);
    const mixer = new AnimationMixer(gltf.scene);
    mixer.clipAction(animation).play();
    const sampleFrameAtTimeS = createFrameSampler({
        scene: gltf.scene,
        mixer,
        records,
    });
    const frames = Array.from(
        { length: manifest.action.cycle_intervals + 1 },
        (_, frameIndex) => sampleFrameAtTimeS(
            frameIndex / manifest.action.fps,
        ),
    );
    const paws = manifest.paws.map(paw => {
        const references = collectSurfaceReferences({
            records,
            meshNames: paw.solid_meshes,
            deformRoot: paw.deform_root,
            weightThreshold: manifest.sdf.surface_weight_threshold,
        });

        const seedIndices = createWeightedPawRegionIndices({
            frame: frames[0],
            meshNames: paw.solid_meshes,
            references,
        });
        const grownIndices = growPawRegionsAboveContactBand({
            frames,
            initialRegionIndices: seedIndices,
            minimumClosureZM: manifest.sdf.minimum_closure_z_m,
        });
        const meshClosures = closeWeightedPawRegions({
            frame: frames[0],
            regionIndices: grownIndices,
            weldToleranceM: manifest.sdf.topology_weld_tolerance_m,
        });

        return {
            manifest: paw,
            references,
            meshIndices: grownIndices,
            meshClosures,
            bounds: measurePawBounds({
                frames,
                references: collectPawRegionReferences(grownIndices),
                paddingM: manifest.sdf.bounds_padding_m,
            }),
        };
    });

    return {
        scene: gltf.scene,
        animation,
        mixer,
        meshNames,
        frames,
        paws,
        sampleFrameAtTimeS,
    };
};
