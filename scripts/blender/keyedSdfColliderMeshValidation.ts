import { validateWatertightTopology } from "./keyedSdfBvh.ts";
import { KEYED_SDF_VALIDATION_THRESHOLDS } from "./keyedSdfValidationThresholds.ts";
import { applyPawMeshClosure } from "./zubeiaPawClosure.ts";
import type {
    KeyedSdfManifest,
    MeshFrame,
    ZubeiaGeometrySource,
} from "./zubeiaKeyedSdfTypes.ts";

type GeometryForMeshValidation = Pick<
    ZubeiaGeometrySource,
    "frames" | "paws"
>;


export const validateOrientedClosedMesh = ({
    frame,
    weldToleranceM,
}: {
    frame: MeshFrame,
    weldToleranceM: number,
}) => {
    const topology = validateWatertightTopology({
        ...frame,
        weldToleranceM,
        minimumTriangleDoubleAreaM2:
            KEYED_SDF_VALIDATION_THRESHOLDS.minimumTriangleDoubleAreaM2,
    });
    const volumeM3 = Math.abs(topology.signedVolumeM3);
    if (
        volumeM3
        < KEYED_SDF_VALIDATION_THRESHOLDS.minimumClosedComponentVolumeM3
    ) {
        throw new Error("closed paw mesh has zero signed volume");
    }
    return {
        ...topology,
        orientationSign: Math.sign(topology.signedVolumeM3) as -1 | 1,
        minimumComponentVolumeM3: volumeM3,
    };
};

const minimumCapZM = (
    sourceFrame: MeshFrame,
    closedFrame: MeshFrame,
    capLoops: readonly (readonly number[])[],
) => {
    const sourceVertexCount = sourceFrame.positions.length / 3;
    return Math.min(...capLoops.flatMap((loop, loopIndex) => [
        ...loop.map(vertexIndex => sourceFrame.positions[vertexIndex * 3 + 2]),
        closedFrame.positions[(sourceVertexCount + loopIndex) * 3 + 2],
    ]));
};

export const validateKeyedSdfColliderMeshes = ({
    geometry,
    manifest,
}: {
    geometry: GeometryForMeshValidation,
    manifest: KeyedSdfManifest,
}) => {
    const expectedFrameCount = manifest.action.cycle_intervals + 1;
    if (geometry.frames.length !== expectedFrameCount) {
        throw new Error(
            `closed paw validation requires ${expectedFrameCount} frames`,
        );
    }

    let minimumTriangleDoubleAreaM2 = Infinity;
    let minimumComponentVolumeM3 = Infinity;
    let capMinimumZM = Infinity;
    let capLoopCount = 0;
    let validatedMeshFrameCount = 0;
    geometry.frames.forEach((frame, frameIndex) => {
        geometry.paws.forEach(paw => {
            paw.meshClosures.forEach((closure, meshName) => {
                const sourceFrame = frame.get(meshName);
                if (sourceFrame === undefined) {
                    throw new Error(
                        `${paw.manifest.name} frame ${frameIndex} has no ${meshName}`,
                    );
                }
                const closedFrame = applyPawMeshClosure({
                    frame: sourceFrame,
                    closure,
                });
                const metric = validateOrientedClosedMesh({
                    frame: closedFrame,
                    weldToleranceM: manifest.sdf.topology_weld_tolerance_m,
                });
                minimumTriangleDoubleAreaM2 = Math.min(
                    minimumTriangleDoubleAreaM2,
                    metric.minimumTriangleDoubleAreaM2,
                );
                minimumComponentVolumeM3 = Math.min(
                    minimumComponentVolumeM3,
                    metric.minimumComponentVolumeM3,
                );
                if (closure.capLoops.length > 0) {
                    capMinimumZM = Math.min(
                        capMinimumZM,
                        minimumCapZM(
                            sourceFrame,
                            closedFrame,
                            closure.capLoops,
                        ),
                    );
                }
                capLoopCount += closure.capLoops.length;
                validatedMeshFrameCount++;
            });
        });
    });

    if (validatedMeshFrameCount === 0) {
        throw new Error("closed paw validation found no meshes");
    }
    const capClearanceM = capLoopCount === 0
        ? null
        : capMinimumZM - manifest.sdf.minimum_closure_z_m;
    if (
        capClearanceM !== null
        && capClearanceM
            < -KEYED_SDF_VALIDATION_THRESHOLDS.capHeightToleranceM
    ) {
        throw new Error(
            `closed paw cap ${capMinimumZM.toFixed(6)} m is below `
            + `${manifest.sdf.minimum_closure_z_m.toFixed(6)} m`,
        );
    }

    return {
        validatedFrameCount: geometry.frames.length,
        validatedMeshFrameCount,
        capLoopCountPerFrame: capLoopCount / geometry.frames.length,
        minimumCapZM: capLoopCount === 0 ? null : capMinimumZM,
        minimumRequiredCapZM: manifest.sdf.minimum_closure_z_m,
        minimumCapClearanceM: capClearanceM,
        minimumTriangleDoubleAreaM2,
        minimumComponentVolumeM3,
        thresholds: KEYED_SDF_VALIDATION_THRESHOLDS,
    };
};
