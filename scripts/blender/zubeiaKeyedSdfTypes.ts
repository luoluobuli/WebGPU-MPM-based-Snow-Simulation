import type { AnimationClip, AnimationMixer, Object3D } from "three";
import type { Vec3 } from "./keyedSdfBvh.ts";
import type { PawMeshClosure } from "./zubeiaPawClosure.ts";

export type PawManifest = {
    name: string,
    deform_root: string,
    solid_meshes: string[],
};

export type KeyedSdfManifest = {
    action: {
        name: string,
        fps: number,
        cycle_intervals: number,
        preview_cycles: number,
        root_motion: "in_place",
    },
    axes: {
        forward: Vec3,
        up: Vec3,
    },
    gait: {
        stride_length: number,
    },
    sdf: {
        bounds_padding_m: number,
        topology_weld_tolerance_m: number,
        minimum_closure_z_m: number,
        surface_weight_threshold: number,
    },
    paws: PawManifest[],
};

export type MeshFrame = {
    positions: Float32Array,
    indices: Uint32Array,
};

export type SdfGridDimensions = readonly [
    number,
    number,
    number,
];

export type PawSurfaceReference = {
    meshName: string,
    vertexIndex: number,
};

export type PawFieldBounds = {
    minLocal: Vec3,
    maxLocal: Vec3,
};

export type PawSource = {
    manifest: PawManifest,
    bounds: PawFieldBounds,
    references: PawSurfaceReference[],
    meshIndices: ReadonlyMap<string, Uint32Array>,
    meshClosures: ReadonlyMap<string, PawMeshClosure>,
};

export type ZubeiaGeometrySource = {
    scene: Object3D,
    animation: AnimationClip,
    mixer: AnimationMixer,
    meshNames: string[],
    frames: Map<string, MeshFrame>[],
    paws: PawSource[],
    sampleFrameAtTimeS: (timeS: number) => Map<string, MeshFrame>,
};

export type PackedPawField = {
    phi: Float32Array,
    velocity: Float32Array,
};
