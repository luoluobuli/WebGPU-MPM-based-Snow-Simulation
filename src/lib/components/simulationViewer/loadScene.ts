// import { load } from "@loaders.gl/core";
// import { GLTFLoader } from "@loaders.gl/gltf";

// export const loadMeshes = async (url: string) => {
//     const {json} = await load(url, GLTFLoader);

//     if (json.meshes === undefined) return;

//     for (const mesh of json.meshes) {

//     }
// };

import {GLTFLoader, type GLTF} from "three/addons/loaders/GLTFLoader.js";
import { Matrix4, Mesh, MeshPhysicalMaterial, Object3D, Vector3 } from "three";


const traverseChildren = (scene: Object3D, fn: (child: Object3D) => void) => {
    fn(scene);

    for (const child of scene.children) {
        traverseChildren(child, fn);
    }
};

const vec = (array: Float32Array, mat: Matrix4) => {
    const vec3 = new Vector3(array[0], array[1], array[2]).applyMatrix4(mat);
    return [vec3.x, vec3.z, vec3.y];
};

export const loadGltfScene = async (url: string) => {
    const gltf = await new Promise<GLTF>((resolve, _reject) => new GLTFLoader().load(url, resolve));

    let nTriBytes = 0;
    let nMaterialBytes = 0;
    let nBoundingBoxBytes = 0;

    const materialMap = new Map<MeshPhysicalMaterial, number>();

    traverseChildren(gltf.scene, child => {
        if (!(child instanceof Mesh)) return;
        nTriBytes += child.geometry.index.array.length / 3 * 48;
        
        if (!materialMap.has(child.material)) {
            materialMap.set(child.material, materialMap.size);
            nMaterialBytes += 48;
        }

        nBoundingBoxBytes += 32;
    });

    const triangles = new ArrayBuffer(nTriBytes);
    let triOffset = 0;

    const boundingBoxes = new ArrayBuffer(nBoundingBoxBytes);
    let boundingBoxOffset = 0;

    traverseChildren(gltf.scene, child => {
        if (!(child instanceof Mesh)) return;


        const boxTriangleIndex = triOffset / 48;
        const boxMin = [Infinity, Infinity, Infinity];
        const boxMax = [-Infinity, -Infinity, -Infinity];


        const pos = child.geometry.attributes.position.array;
        const index = child.geometry.index.array;

        for (let i = 0; i < index.length; i += 3) {
            const v0 = vec(pos.slice(3 * index[i], 3 * index[i] + 3), child.matrix);
            const v1 = vec(pos.slice(3 * index[i + 1], 3 * index[i + 1] + 3), child.matrix);
            const v2 = vec(pos.slice(3 * index[i + 2], 3 * index[i + 2] + 3), child.matrix);

            new Float32Array(triangles, triOffset).set(v0);
            new Float32Array(triangles, triOffset + 16).set(v1);
            new Float32Array(triangles, triOffset + 32).set(v2);
            new Uint32Array(triangles, triOffset + 12).set([materialMap.get(child.material)!]);

            boxMin[0] = Math.min(boxMin[0], v0[0], v1[0], v2[0]);
            boxMin[1] = Math.min(boxMin[1], v0[1], v1[1], v2[1]);
            boxMin[2] = Math.min(boxMin[2], v0[2], v1[2], v2[2]);
            boxMax[0] = Math.max(boxMax[0], v0[0], v1[0], v2[0]);
            boxMax[1] = Math.max(boxMax[1], v0[1], v1[1], v2[1]);
            boxMax[2] = Math.max(boxMax[2], v0[2], v1[2], v2[2]);

            triOffset += 48;
        }

        new Float32Array(boundingBoxes, boundingBoxOffset).set([
            boxMin[0], boxMin[1], boxMin[2], 0,
            boxMax[0], boxMax[1], boxMax[2],
        ]);
        new Uint32Array(boundingBoxes, boundingBoxOffset + 28).set([boxTriangleIndex]);

        boundingBoxOffset += 32;
    });

    const materials = new ArrayBuffer(nMaterialBytes);
    for (const [material, i] of materialMap) {
        new Float32Array(materials, i * 48).set([
            material.color.r,
            material.color.g,
            material.color.b,
            Object.hasOwn(material, "transmission") ? 1 - material.transmission : 1,

            material.emissive.r * material.emissiveIntensity,
            material.emissive.g * material.emissiveIntensity,
            material.emissive.b * material.emissiveIntensity,
            [material.emissiveIntensity, material.emissive.r, material.emissive.g, material.emissive.b].every(c => c > 0) ? 1 : 0,

            material.roughness,
            0,
            0,
            0,
        ]);
    }

    const vertices: number[][] = [];
    const positions: number[] = [];
    const indices: number[] = [];
    var vertexOffset = 0;

    traverseChildren(gltf.scene, child => {
        if (!(child instanceof Mesh)) return;
        
        const pos = child.geometry.attributes.position.array;
        const index = child.geometry.index.array;
        
        for (let i = 0; i < index.length; i++) {
            const v = vec(pos.slice(3 * index[i], 3 * index[i] + 3), child.matrix);
            vertices.push(v);
            indices.push(index[i] + vertexOffset);
        }

        for (let i = 0; i < pos.length; i+= 3) {
            const px = pos[i];
            const py = pos[i + 1];
            const pz = pos[i + 2];

            const v = vec(new Float32Array([px, py, pz]), child.matrix);
            positions.push(v[0], v[1], v[2]);
        }
        vertexOffset += pos.length / 3;

    });

    return {
        boundingBoxes,
        triangles,
        materials,
        vertices,
        positions,
        indices,
    };
};