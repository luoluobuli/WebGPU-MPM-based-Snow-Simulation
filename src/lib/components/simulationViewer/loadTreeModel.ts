import { GLTFLoader, type GLTF } from "three/addons/loaders/GLTFLoader.js";
import { Mesh, Object3D, Vector3 } from "three";
import type { BufferAttribute, InterleavedBufferAttribute, Matrix4 } from "three";
import {
    buildTreeModelSpawnPoints,
    type TreeModelMesh,
    type TreeModelSpawnPoints,
    type Vec3,
} from "./treeModelSpawnPoints";

type PositionAttribute = BufferAttribute | InterleavedBufferAttribute;

const traverseChildren = (scene: Object3D, fn: (child: Object3D) => void) => {
    fn(scene);

    for (const child of scene.children) {
        traverseChildren(child, fn);
    }
};

const readVertex = (
    position: PositionAttribute,
    index: number,
    mat: Matrix4,
): Vec3 => {
    const vertex = new Vector3(
        position.getX(index),
        position.getY(index),
        position.getZ(index),
    ).applyMatrix4(mat);

    return [vertex.x, vertex.z, vertex.y];
};

const meshName = (mesh: Mesh) =>
    [
        mesh.name,
        mesh.geometry.name,
        Array.isArray(mesh.material) ? "" : mesh.material?.name,
    ]
        .filter((name): name is string => typeof name === "string" && name.length > 0)
        .join(" ");

const loadGltf = (url: string) =>
    new Promise<GLTF>((resolve, reject) => {
        new GLTFLoader().load(url, resolve, undefined, reject);
    });

export const collectTreeModelMeshes = (gltf: GLTF) => {
    const meshes: TreeModelMesh[] = [];
    gltf.scene.updateMatrixWorld(true);

    traverseChildren(gltf.scene, (child) => {
        if (!(child instanceof Mesh)) return;

        const position = child.geometry.attributes.position;
        if (position === undefined || position.itemSize < 3) return;

        const vertices: Vec3[] = [];
        const index = child.geometry.index;

        if (index !== null) {
            for (let i = 0; i + 2 < index.count; i += 3) {
                vertices.push(
                    readVertex(position, index.getX(i), child.matrixWorld),
                    readVertex(position, index.getX(i + 1), child.matrixWorld),
                    readVertex(position, index.getX(i + 2), child.matrixWorld),
                );
            }
        } else {
            for (let i = 0; i + 2 < position.count; i += 3) {
                vertices.push(
                    readVertex(position, i, child.matrixWorld),
                    readVertex(position, i + 1, child.matrixWorld),
                    readVertex(position, i + 2, child.matrixWorld),
                );
            }
        }

        if (vertices.length > 0) {
            meshes.push({
                name: meshName(child),
                vertices,
            });
        }
    });

    return meshes;
};

export const loadTreeModelSpawnPoints = async ({
    url,
    nParticles,
}: {
    url: string,
    nParticles: number,
}): Promise<TreeModelSpawnPoints> => {
    const gltf = await loadGltf(url);
    const meshes = collectTreeModelMeshes(gltf);

    return buildTreeModelSpawnPoints({
        meshes,
        nParticles,
    });
};
