// import { load } from "@loaders.gl/core";
// import { GLTFLoader } from "@loaders.gl/gltf";

// export const loadMeshes = async (url: string) => {
//     const {json} = await load(url, GLTFLoader);

//     if (json.meshes === undefined) return;

//     for (const mesh of json.meshes) {

//     }
// };

import {GLTFLoader, type GLTF} from "three/addons/loaders/GLTFLoader.js";
import * as THREE from "three";
import { Matrix4, Matrix3, Mesh, MeshPhysicalMaterial, MeshStandardMaterial, Object3D, Vector3 } from "three";


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
    gltf.scene.updateMatrixWorld(true);

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
            const v0 = vec(pos.slice(3 * index[i], 3 * index[i] + 3), child.matrixWorld);
            const v1 = vec(pos.slice(3 * index[i + 1], 3 * index[i + 1] + 3), child.matrixWorld);
            const v2 = vec(pos.slice(3 * index[i + 2], 3 * index[i + 2] + 3), child.matrixWorld);

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
    const normals: number[] = [];
    const uvs: number[] = [];
    const materialIndices: number[] = [];
    const indices: number[] = [];
    
    const textureMap = new Map<THREE.Texture, number>();
    const textures: THREE.Texture[] = [];
    
    // Track objects for broadphase collision
    const objects: {
        min: [number, number, number];
        max: [number, number, number];
        startIndex: number;
        countIndices: number;
    }[] = [];

    var vertexOffset = 0;

    traverseChildren(gltf.scene, child => {
        if (!(child instanceof Mesh)) return;
        
        const pos_in = child.geometry.attributes.position.array;
        const nor_in = child.geometry.attributes.normal?.array;
        const uv_in = child.geometry.attributes.uv?.array;
        const idx_in = child.geometry.index.array;
        
        const material = child.material as MeshStandardMaterial;
        const texture = material?.map;
        
        let materialIndex = 0;
        if (texture) {
            if (!textureMap.has(texture)) {
                textureMap.set(texture, textures.length);
                textures.push(texture);
            }
            materialIndex = textureMap.get(texture)!;
        }
        
        const startIndex = indices.length;
        const objectMin: [number, number, number] = [Infinity, Infinity, Infinity];
        const objectMax: [number, number, number] = [-Infinity, -Infinity, -Infinity];

        // indices
        for (let i = 0; i < idx_in.length; i++) {
            const v = vec(pos_in.slice(3 * idx_in[i], 3 * idx_in[i] + 3), child.matrixWorld);
            
            objectMin[0] = Math.min(objectMin[0], v[0]);
            objectMin[1] = Math.min(objectMin[1], v[1]);
            objectMin[2] = Math.min(objectMin[2], v[2]);
            objectMax[0] = Math.max(objectMax[0], v[0]);
            objectMax[1] = Math.max(objectMax[1], v[1]);
            objectMax[2] = Math.max(objectMax[2], v[2]);

            vertices.push(v);
            
            indices.push(idx_in[i] + vertexOffset);
        }

        const invTransMat = new Matrix3().getNormalMatrix(child.matrixWorld);

        // flat vertex attributes
        for (let i = 0; i < pos_in.length; i+= 3) {
            // positions
            const p = vec(new Float32Array([pos_in[i], pos_in[i+1], pos_in[i+2]]), child.matrixWorld);
            positions.push(p[0], p[1], p[2]);

            // normals
            if (nor_in) {
                const n = new Vector3(nor_in[i], nor_in[i+1], nor_in[i+2]).applyMatrix3(invTransMat).normalize();
                normals.push(n.x, n.y, n.z);
            }

            // uvs
            const vertIdx = i / 3;
            if (uv_in) {
                uvs.push(uv_in[vertIdx * 2], uv_in[vertIdx * 2 + 1]);
            } else {
                uvs.push(0, 0);
            }
            
            materialIndices.push(materialIndex);
        }
        vertexOffset += pos_in.length / 3;

        objects.push({
            min: objectMin,
            max: objectMax,
            startIndex: startIndex,
            countIndices: indices.length - startIndex,
        });
    });
    

    const textureBitmaps: ImageBitmap[] = [];
    for (const texture of textures) {
        if (texture.image) {
            try {
                const bitmap = await createImageBitmap(texture.image as ImageBitmapSource);
                textureBitmaps.push(bitmap);
            } catch (e) {
                console.warn("Failed to convert texture to ImageBitmap:", e);
                const canvas = new OffscreenCanvas(1, 1);
                const ctx = canvas.getContext("2d")!;
                ctx.fillStyle = "white";
                ctx.fillRect(0, 0, 1, 1);
                const bitmap = await createImageBitmap(canvas);
                textureBitmaps.push(bitmap);
            }
        } else {
            const canvas = new OffscreenCanvas(1, 1);
            const ctx = canvas.getContext("2d")!;
            ctx.fillStyle = "white";
            ctx.fillRect(0, 0, 1, 1);
            const bitmap = await createImageBitmap(canvas);
            textureBitmaps.push(bitmap);
        }
    }

    return {
        materials,
        vertices,
        positions,
        normals,
        uvs,
        materialIndices,
        textures: textureBitmaps,
        indices,
        objects,
    };
};