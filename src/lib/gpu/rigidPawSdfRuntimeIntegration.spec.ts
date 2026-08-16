import { describe, expect, it } from "vitest";

import sceneSrc from "../components/simulationViewer/SimulationScene.ts?raw";
import stateSrc from "../components/simulationViewer/SimulationState.svelte.ts?raw";
import loaderSrc from "../components/simulationViewer/zubeia/loadZubeiaRigidPawWalkCollider.ts?raw";
import runnerSrc from "./GpuSnowPipelineRunner.svelte.ts?raw";
import managerSrc from "./kinematicCollider/GpuRigidPawSdfKinematicColliderManager.ts?raw";
import colliderSrc from "./kinematicCollider/rigidPawSdfKinematicColliderPrelude.wgsl?raw";
import pipelineSrc from "./mpm/GpuMpmPipelineManager.ts?raw";

const sourceBetween = (
    source: string,
    startMarker: string,
    endMarker: string,
) => {
    const start = source.indexOf(startMarker);
    const end = source.indexOf(endMarker, start);

    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);
    return source.slice(start, end);
};

describe("rigid paw SDF runtime integration", () => {
    it("loads one static field per actual animated paw", () => {
        expect(sceneSrc).toContain("zubeia-paw-sdf.bin?url");
        expect(sceneSrc).toContain("rigidPawSdfUrl");
        expect(stateSrc).toContain(
            "rigidPawSdfUrl: scene.colliderSource.rigidPawSdfUrl",
        );
        expect(loaderSrc).toContain(
            "loadRigidPawSdfColliderAsset",
        );
        expect(loaderSrc).toContain(
            "sampleWorldFromUnitAtTimeS",
        );
        expect(loaderSrc).toContain("bones[pawIndex].matrixWorld");
        expect(loaderSrc).not.toContain("loadKeyedSdfColliderClip");
    });

    it("uploads static phi once and drains bounded substep batches", () => {
        const method = sourceBetween(
            runnerSrc,
            "    private async advanceKinematicColliderSubsteps({",
            "    private mapSubmittedParticleSpeed()",
        );
        const intervalUpload = method.indexOf("writeIntervals({");
        const submissionSchedule = method.indexOf(
            "calculateKinematicColliderSubmissionRanges({",
        );
        const submissionWindow = method.indexOf(
            "submitKinematicColliderSubmissionRangesInWindows({",
        );
        const submissionCallback = method.indexOf(
            "submitRange: ({ startSubstep, endSubstep }) => {",
        );
        const substepLoop = method.indexOf(
            "let substep = startSubstep;",
        );
        const intervalCopy = method.indexOf("copyIntervalTo({");
        const simulationStep = method.indexOf(
            "this.addSimulationStepsComputePass({",
        );
        const submission = method.indexOf(
            "this.device.queue.submit([commandEncoder.finish()])",
        );
        const queueDrain = method.indexOf(
            "this.device.queue.onSubmittedWorkDone()",
        );

        expect(runnerSrc).toContain(
            "new GpuRigidPawSdfKinematicColliderManager({",
        );
        expect(managerSrc).toContain(
            "device.queue.writeBuffer(sdfBuffer, 0, sdfUpload)",
        );
        expect(method).not.toContain(
            "writeResidentFramesForOutputFrame",
        );
        expect(intervalUpload).toBeGreaterThanOrEqual(0);
        expect(intervalUpload).toBeLessThan(submissionSchedule);
        expect(submissionSchedule).toBeLessThan(submissionWindow);
        expect(submissionWindow).toBeLessThan(submissionCallback);
        expect(submissionCallback).toBeLessThan(substepLoop);
        expect(intervalCopy).toBeGreaterThan(substepLoop);
        expect(intervalCopy).toBeLessThan(simulationStep);
        expect(submission).toBeGreaterThan(simulationStep);
        expect(queueDrain).toBeGreaterThan(submission);
        expect(method).toContain(
            "const isLastSubstep = substep === nSubsteps - 1",
        );
    });

    it("derives rigid surface velocity from adjacent transforms", () => {
        expect(colliderSrc).toContain(
            "previousWorldFromUnit: mat4x4f",
        );
        expect(colliderSrc).toContain(
            "transformedCurrentSurface",
        );
        expect(colliderSrc).toContain(
            "transformedPreviousSurface",
        );
        expect(colliderSrc).toContain(
            "relativeVelocity = velocity - sample.surfaceVelocity",
        );
        expect(colliderSrc).not.toContain(
            "sampleKinematicColliderAnimatedField",
        );
    });

    it("keeps post-advection recovery position-only", () => {
        expect(colliderSrc).toContain(
            "resolveParticlePositionAgainstKinematicCollider",
        );
        expect(colliderSrc).toContain(
            "- sample.signedDistance * sample.normal",
        );
    });

    it("binds and specializes both rigid paw resources", () => {
        expect(pipelineSrc).toContain(
            "../kinematicCollider/rigidPawSdfKinematicColliderPrelude.wgsl?raw",
        );
        expect(pipelineSrc).toContain(
            "{ binding: 12, resource: { buffer: kinematicColliderManager.sdfBuffer } }",
        );
        expect(pipelineSrc).toMatch(
            /KINEMATIC_COLLIDER_SDF_RESOLUTION:\r?\n\s+kinematicColliderManager\.sdfResolution/,
        );
    });
});
