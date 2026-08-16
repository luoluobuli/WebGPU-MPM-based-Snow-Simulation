import {
    ZUBEIA_PAW_COLLIDER_DEFINITIONS,
    ZUBEIA_PAW_SDF_MAX_CELL_SIZE_M,
    ZUBEIA_PAW_SDF_RESOLUTION,
} from "../../src/lib/components/simulationViewer/zubeia/ZubeiaPawColliderCalibration.ts";
import {
    ZUBEIA_WALK_ROOT_PITCH_RAD,
} from "../../src/lib/components/simulationViewer/zubeia/ZubeiaWalkCalibration.ts";
import {
    encodeRigidPawSdfColliderAsset,
    type RigidPawSdfColliderAssetSource,
} from "../../src/lib/gpu/collider/rigidPawSdfColliderAsset.ts";
import {
    Matrix4,
    PropertyBinding,
    Vector3,
    type Object3D,
} from "three";
import type { Vec3 } from "./keyedSdfBvh.ts";
import {
    bakePawField,
    createFrameBvhs,
    float32ToFloat16Bits,
} from "./keyedSdfField.ts";
import { measureTightPawFieldBounds } from "./keyedSdfFrameBounds.ts";
import { float16BitsToFloat32 } from "./keyedSdfPacking.ts";
import { validateKeyedSdfColliderMeshes } from "./keyedSdfColliderMeshValidation.ts";
import { loadZubeiaKeyedSdfGeometry } from "./zubeiaKeyedSdfGeometry.ts";
import type {
    KeyedSdfManifest,
    MeshFrame,
    PawSource,
} from "./zubeiaKeyedSdfTypes.ts";

const sourceUrl = new URL(
    "../../src/lib/assets/models/zubeia-walk-lod.glb",
    import.meta.url,
);
const manifestUrl = new URL(
    "zubeia_keyed_sdf_manifest.json",
    import.meta.url,
);
const outputUrl = new URL(
    "../../src/lib/assets/models/zubeia-paw-sdf.bin",
    import.meta.url,
);
const metricsUrl = new URL(
    "../../src/lib/assets/models/zubeia-paw-sdf.metrics.json",
    import.meta.url,
);

const checksum = (bytes: Uint8Array) => {
    let hash = 0x811c9dc5;
    bytes.forEach(byte => {
        hash ^= byte;
        hash = Math.imul(hash, 0x01000193);
    });

    return (hash >>> 0).toString(16).padStart(8, "0");
};

const bytesEqual = (first: Uint8Array, second: Uint8Array) =>
    first.length === second.length
    && first.every((value, index) => value === second[index]);

const quantizePhiInPlace = (phi: Float32Array) => {
    phi.forEach((value, index) => {
        phi[index] = float16BitsToFloat32(
            float32ToFloat16Bits(value),
        );
    });
};

const packPhiFields = (fields: readonly Float32Array[]) => {
    const voxelCount = fields.reduce(
        (count, field) => count + field.length,
        0,
    );
    const packed = new Uint32Array(Math.ceil(voxelCount / 2));
    let globalVoxel = 0;

    fields.forEach(field => {
        field.forEach(value => {
            const bits = float32ToFloat16Bits(value);
            const wordIndex = Math.floor(globalVoxel / 2);
            if ((globalVoxel & 1) === 0) {
                packed[wordIndex] = bits;
            } else {
                packed[wordIndex] |= bits << 16;
            }
            globalVoxel++;
        });
    });

    return packed;
};

const minimumRegionZ = (
    frame: ReadonlyMap<string, MeshFrame>,
    paw: PawSource,
) => {
    let minimum = Infinity;

    paw.meshIndices.forEach((indices, meshName) => {
        const positions = frame.get(meshName)!.positions;
        indices.forEach(vertexIndex => {
            minimum = Math.min(
                minimum,
                positions[vertexIndex * 3 + 2],
            );
        });
    });

    return minimum;
};

const createSimFromThreeMatrix = () => {
    const cosine = Math.cos(ZUBEIA_WALK_ROOT_PITCH_RAD);
    const sine = Math.sin(ZUBEIA_WALK_ROOT_PITCH_RAD);

    // Three is (x, y, z); simulation space is pitch * (x, z, y).
    return new Matrix4().set(
        1, 0, 0, 0,
        0, -sine, cosine, 0,
        0, cosine, sine, 0,
        0, 0, 0, 1,
    );
};

const findMotionBone = (
    scene: Object3D,
    name: string,
) => {
    const sanitizedName = PropertyBinding.sanitizeNodeName(name);
    const bone = scene.getObjectByName(sanitizedName);
    if (bone === undefined) {
        throw new Error(`Zubeia has no motion bone ${name}`);
    }

    return bone;
};

const sampleBoneWorldMatrix = ({
    geometry,
    bone,
    timeS,
}: {
    geometry: Awaited<
        ReturnType<typeof loadZubeiaKeyedSdfGeometry>
    >,
    bone: Object3D,
    timeS: number,
}) => {
    geometry.mixer.setTime(timeS);
    geometry.scene.updateMatrixWorld(true);

    return bone.matrixWorld.clone();
};

const relativeBoneMatrixInSimSpace = ({
    simFromThree,
    threeFromSim,
    boneWorld,
    referenceBoneWorldInverse,
}: {
    simFromThree: Matrix4,
    threeFromSim: Matrix4,
    boneWorld: Matrix4,
    referenceBoneWorldInverse: Matrix4,
}) => simFromThree.clone()
    .multiply(boneWorld)
    .multiply(referenceBoneWorldInverse)
    .multiply(threeFromSim);

const minimumTransformedReferenceZ = ({
    referenceFrame,
    paw,
    relativeMatrix,
}: {
    referenceFrame: ReadonlyMap<string, MeshFrame>,
    paw: PawSource,
    relativeMatrix: Matrix4,
}) => {
    const point = new Vector3();
    let minimum = Infinity;

    paw.meshIndices.forEach((indices, meshName) => {
        const positions = referenceFrame.get(meshName)!.positions;
        indices.forEach(vertexIndex => {
            const offset = vertexIndex * 3;
            point.set(
                positions[offset],
                positions[offset + 1],
                positions[offset + 2],
            ).applyMatrix4(relativeMatrix);
            minimum = Math.min(minimum, point.z);
        });
    });

    return minimum;
};

const measureVerticalCorrections = ({
    geometry,
    manifest,
}: {
    geometry: Awaited<
        ReturnType<typeof loadZubeiaKeyedSdfGeometry>
    >,
    manifest: KeyedSdfManifest,
}) => {
    const frameCount = manifest.action.cycle_intervals + 1;
    const corrections = new Float32Array(
        frameCount * geometry.paws.length,
    );
    const simFromThree = createSimFromThreeMatrix();
    const threeFromSim = simFromThree.clone().invert();
    const bones = ZUBEIA_PAW_COLLIDER_DEFINITIONS.map(definition =>
        findMotionBone(geometry.scene, definition.motionBoneName)
    );
    const referenceBoneWorldInverses =
        ZUBEIA_PAW_COLLIDER_DEFINITIONS.map(
            (definition, pawIndex) => sampleBoneWorldMatrix({
                geometry,
                bone: bones[pawIndex],
                timeS: definition.referenceFrameIndex
                    / manifest.action.fps,
            }).invert(),
        );
    let maximumAlignmentErrorM = 0;

    for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
        const timeS = frameIndex / manifest.action.fps;

        geometry.paws.forEach((paw, pawIndex) => {
            const definition =
                ZUBEIA_PAW_COLLIDER_DEFINITIONS[pawIndex];
            const referenceFrame =
                geometry.frames[definition.referenceFrameIndex];
            const relativeMatrix = relativeBoneMatrixInSimSpace({
                simFromThree,
                threeFromSim,
                boneWorld: sampleBoneWorldMatrix({
                    geometry,
                    bone: bones[pawIndex],
                    timeS,
                }),
                referenceBoneWorldInverse:
                    referenceBoneWorldInverses[pawIndex],
            });
            const actualMinimumZ = minimumRegionZ(
                geometry.frames[frameIndex],
                paw,
            );
            const transformedMinimumZ =
                minimumTransformedReferenceZ({
                    referenceFrame,
                    paw,
                    relativeMatrix,
                });
            const correction = actualMinimumZ - transformedMinimumZ;
            corrections[
                frameIndex * geometry.paws.length + pawIndex
            ] = correction;
            maximumAlignmentErrorM = Math.max(
                maximumAlignmentErrorM,
                Math.abs(
                    transformedMinimumZ + correction - actualMinimumZ,
                ),
            );
        });
    }

    return {
        corrections,
        maximumAlignmentErrorM,
    };
};

const fieldVoxel = (
    phi: Float32Array,
    resolution: number,
    x: number,
    y: number,
    z: number,
) => phi[x + y * resolution + z * resolution ** 2];

const sampleFieldPhi = ({
    phi,
    bounds,
    point,
}: {
    phi: Float32Array,
    bounds: PawSource["bounds"],
    point: Vec3,
}) => {
    const resolution = ZUBEIA_PAW_SDF_RESOLUTION;
    const last = resolution - 1;
    const grid = point.map((value, axis) =>
        (value - bounds.minLocal[axis])
        / (bounds.maxLocal[axis] - bounds.minLocal[axis])
        * last
    );
    if (grid.some(value => value < 0 || value > last)) {
        return Infinity;
    }

    const base = grid.map(
        value => Math.min(Math.floor(value), last - 1),
    );
    const interpolation = grid.map(
        (value, axis) => value - base[axis],
    );
    const values: number[] = [];
    for (let z = 0; z <= 1; z++) {
        for (let y = 0; y <= 1; y++) {
            for (let x = 0; x <= 1; x++) {
                values.push(fieldVoxel(
                    phi,
                    resolution,
                    base[0] + x,
                    base[1] + y,
                    base[2] + z,
                ));
            }
        }
    }

    const x00 = values[0] * (1 - interpolation[0])
        + values[1] * interpolation[0];
    const x10 = values[2] * (1 - interpolation[0])
        + values[3] * interpolation[0];
    const x01 = values[4] * (1 - interpolation[0])
        + values[5] * interpolation[0];
    const x11 = values[6] * (1 - interpolation[0])
        + values[7] * interpolation[0];
    const y0 = x00 * (1 - interpolation[1])
        + x10 * interpolation[1];
    const y1 = x01 * (1 - interpolation[1])
        + x11 * interpolation[1];

    return y0 * (1 - interpolation[2])
        + y1 * interpolation[2];
};

const validateStaticField = ({
    phi,
    paw,
    frame,
    cellSizeM,
    negativeVoxelCount,
}: {
    phi: Float32Array,
    paw: PawSource,
    frame: ReadonlyMap<string, MeshFrame>,
    cellSizeM: Vec3,
    negativeVoxelCount: number,
}) => {
    const resolution = ZUBEIA_PAW_SDF_RESOLUTION;
    let negativeBoundaryVoxelCount = 0;
    let maximumVisibleSurfaceMismatchM = 0;
    const visitedVertices = new Set<string>();

    for (let z = 0; z < resolution; z++) {
        for (let y = 0; y < resolution; y++) {
            for (let x = 0; x < resolution; x++) {
                if (
                    x !== 0
                    && y !== 0
                    && z !== 0
                    && x !== resolution - 1
                    && y !== resolution - 1
                    && z !== resolution - 1
                ) {
                    continue;
                }
                negativeBoundaryVoxelCount += Number(
                    fieldVoxel(phi, resolution, x, y, z) < 0,
                );
            }
        }
    }

    paw.meshIndices.forEach((indices, meshName) => {
        const positions = frame.get(meshName)!.positions;
        indices.forEach(vertexIndex => {
            const key = `${meshName}:${vertexIndex}`;
            if (visitedVertices.has(key)) return;
            visitedVertices.add(key);

            const offset = vertexIndex * 3;
            const point: Vec3 = [
                positions[offset],
                positions[offset + 1],
                positions[offset + 2],
            ];
            maximumVisibleSurfaceMismatchM = Math.max(
                maximumVisibleSurfaceMismatchM,
                Math.abs(sampleFieldPhi({
                    phi,
                    bounds: paw.bounds,
                    point,
                })),
            );
        });
    });

    const maximumCellSizeM = Math.max(...cellSizeM);
    if (maximumCellSizeM > ZUBEIA_PAW_SDF_MAX_CELL_SIZE_M) {
        throw new Error(
            `${paw.manifest.name} SDF cell ${maximumCellSizeM} m exceeds `
            + `${ZUBEIA_PAW_SDF_MAX_CELL_SIZE_M} m`,
        );
    }
    if (negativeVoxelCount === 0) {
        throw new Error(`${paw.manifest.name} has no interior voxels`);
    }
    if (negativeBoundaryVoxelCount !== 0) {
        throw new Error(
            `${paw.manifest.name} reaches the SDF field boundary`,
        );
    }
    if (maximumVisibleSurfaceMismatchM > maximumCellSizeM * 1.5) {
        throw new Error(
            `${paw.manifest.name} surface mismatch `
            + `${maximumVisibleSurfaceMismatchM} m exceeds 1.5 cells`,
        );
    }

    return {
        cellSizeM,
        maximumCellSizeM,
        negativeVoxelCount,
        negativeBoundaryVoxelCount,
        visibleSurfaceSampleCount: visitedVertices.size,
        maximumVisibleSurfaceMismatchM,
    };
};

const buildAsset = async () => {
    const manifest = JSON.parse(
        await Deno.readTextFile(manifestUrl),
    ) as KeyedSdfManifest;
    const geometry = await loadZubeiaKeyedSdfGeometry({
        sourceUrl,
        manifest,
    });
    if (
        geometry.paws.length
            !== ZUBEIA_PAW_COLLIDER_DEFINITIONS.length
        || geometry.paws.some(
            (paw, index) =>
                paw.manifest.name
                !== ZUBEIA_PAW_COLLIDER_DEFINITIONS[index].name,
        )
    ) {
        throw new Error(
            "paw manifest order differs from collider definitions",
        );
    }

    const cappedPawMeshValidation = validateKeyedSdfColliderMeshes({
        geometry,
        manifest,
    });
    const fields: Float32Array[] = [];
    const volumes: RigidPawSdfColliderAssetSource["volumes"] = [];
    const fieldMetrics = [];

    for (
        let pawIndex = 0;
        pawIndex < geometry.paws.length;
        pawIndex++
    ) {
        const sourcePaw = geometry.paws[pawIndex];
        const definition = ZUBEIA_PAW_COLLIDER_DEFINITIONS[pawIndex];
        const frameIndex = definition.referenceFrameIndex;
        const frame = geometry.frames[frameIndex];
        const paw: PawSource = {
            ...sourcePaw,
            bounds: measureTightPawFieldBounds({
                frame,
                paw: sourcePaw,
                paddingM: manifest.sdf.bounds_padding_m,
            }),
        };
        const previousFrameIndex = (
            frameIndex
            + manifest.action.cycle_intervals
            - 1
        ) % manifest.action.cycle_intervals;
        const nextFrameIndex =
            (frameIndex + 1) % manifest.action.cycle_intervals;
        const baked = bakePawField({
            paw,
            resolution: ZUBEIA_PAW_SDF_RESOLUTION,
            previousFrame: geometry.frames[previousFrameIndex],
            nextFrame: geometry.frames[nextFrameIndex],
            currentBvhs: createFrameBvhs(
                frame,
                paw.meshClosures,
            ),
            sampleRateHz: manifest.action.fps,
        });
        quantizePhiInPlace(baked.field.phi);
        fields.push(baked.field.phi);
        volumes.push({
            minReference: [...paw.bounds.minLocal] as Vec3,
            maxReference: [...paw.bounds.maxLocal] as Vec3,
            referenceFrameIndex: frameIndex,
        });
        fieldMetrics.push({
            name: definition.name,
            motionBoneName: definition.motionBoneName,
            referenceFrameIndex: frameIndex,
            referenceTimeS: frameIndex / manifest.action.fps,
            ...validateStaticField({
                phi: baked.field.phi,
                paw,
                frame,
                cellSizeM: baked.cellSize,
                negativeVoxelCount: baked.negativeVoxelCount,
            }),
        });
        console.log(
            `baked rigid paw SDF ${pawIndex + 1}/`
            + `${geometry.paws.length}: ${definition.name}`,
        );
    }

    const {
        corrections,
        maximumAlignmentErrorM,
    } = measureVerticalCorrections({
        geometry,
        manifest,
    });
    const cycleDurationS = manifest.action.cycle_intervals
        / manifest.action.fps;
    const source: RigidPawSdfColliderAssetSource = {
        cycleCount: manifest.action.preview_cycles,
        sdfResolution: ZUBEIA_PAW_SDF_RESOLUTION,
        sampleRateHz: manifest.action.fps,
        cycleDurationS,
        durationS: cycleDurationS
            * manifest.action.preview_cycles,
        stridePerCycle: [0, manifest.gait.stride_length, 0],
        volumes,
        verticalCorrectionM: corrections,
        packedPhi: packPhiFields(fields),
    };
    const bytes = encodeRigidPawSdfColliderAsset(source);
    const metrics = {
        source: "actual capped visible skinned Zubeia paw topology",
        representation:
            "four stationary metric SDFs with animated rigid transforms",
        collisionMethod:
            "Disney 2013 rigid level set, grid and particle Coulomb projection",
        pawCount: geometry.paws.length,
        sdfResolution: ZUBEIA_PAW_SDF_RESOLUTION,
        packedChannelLayout: "two float16 metric phi values per u32",
        assetByteLength: bytes.byteLength,
        maximumAllowedCellSizeM: ZUBEIA_PAW_SDF_MAX_CELL_SIZE_M,
        maximumSoleAlignmentErrorM: maximumAlignmentErrorM,
        fieldMetrics,
        cappedPawMeshValidation,
        fnv1a: checksum(bytes),
    };

    return {
        bytes,
        metricsText: `${JSON.stringify(metrics, null, 4)}\n`,
        summary: metrics,
    };
};

const checkOnly = Deno.args.includes("--check");
if (Deno.args.some(argument => argument !== "--check")) {
    throw new Error(
        "usage: deno run --allow-read --allow-write "
        + "<script> [--check]",
    );
}

const {
    bytes,
    metricsText,
    summary,
} = await buildAsset();
if (checkOnly) {
    const [existingBytes, existingMetrics] = await Promise.all([
        Deno.readFile(outputUrl),
        Deno.readTextFile(metricsUrl),
    ]);
    if (!bytesEqual(bytes, existingBytes)) {
        throw new Error(
            "zubeia-paw-sdf.bin is stale; rerun the baker",
        );
    }
    if (existingMetrics !== metricsText) {
        throw new Error(
            "zubeia-paw-sdf.metrics.json is stale; rerun the baker",
        );
    }
} else {
    await Promise.all([
        Deno.writeFile(outputUrl, bytes),
        Deno.writeTextFile(metricsUrl, metricsText),
    ]);
}

console.log(
    `${checkOnly ? "verified" : "wrote"} ${bytes.byteLength} bytes; `
    + `max cell ${Math.max(
        ...summary.fieldMetrics.map(metric =>
            metric.maximumCellSizeM
        ),
    ).toFixed(6)} m; fnv1a ${summary.fnv1a}`,
);
