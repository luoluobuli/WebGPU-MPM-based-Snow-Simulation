import { describe, expect, it } from "vitest";

import {
    calculateKinematicColliderSubmissionRanges,
    MAX_KINEMATIC_COLLIDER_PARTICLE_SUBSTEPS_PER_SUBMISSION,
    MAX_KINEMATIC_COLLIDER_SUBMISSIONS_PER_WINDOW,
    submitKinematicColliderSubmissionRangesInWindows,
} from "./kinematicColliderSubmissionSchedule";

const expectCompleteSchedule = ({
    nSubsteps,
    nParticles,
    expectedSubstepsPerSubmission,
}: {
    nSubsteps: number,
    nParticles: number,
    expectedSubstepsPerSubmission: number,
}) => {
    const ranges = calculateKinematicColliderSubmissionRanges({
        nSubsteps,
        nParticles,
    });

    expect(ranges[0]?.startSubstep).toBe(0);
    expect(ranges.at(-1)?.endSubstep).toBe(nSubsteps);

    for (let rangeIndex = 0; rangeIndex < ranges.length; rangeIndex++) {
        const range = ranges[rangeIndex];
        const rangeSubstepCount =
            range.endSubstep - range.startSubstep;

        expect(rangeSubstepCount).toBeGreaterThan(0);
        expect(rangeSubstepCount).toBeLessThanOrEqual(
            expectedSubstepsPerSubmission,
        );
        if (rangeIndex > 0) {
            expect(range.startSubstep).toBe(
                ranges[rangeIndex - 1].endSubstep,
            );
        }
    }

    return ranges;
};

describe("kinematic collider submission schedule", () => {
    it("keeps preview frames in four contiguous submissions", () => {
        const ranges = expectCompleteSchedule({
            nSubsteps: 112,
            nParticles: 131_648,
            expectedSubstepsPerSubmission: 32,
        });

        expect(ranges).toEqual([
            { startSubstep: 0, endSubstep: 32 },
            { startSubstep: 32, endSubstep: 64 },
            { startSubstep: 64, endSubstep: 96 },
            { startSubstep: 96, endSubstep: 112 },
        ]);
    });

    it("scales the batch size down for two and four particles per cell", () => {
        const twoParticleRanges = expectCompleteSchedule({
            nSubsteps: 112,
            nParticles: 263_296,
            expectedSubstepsPerSubmission: 16,
        });
        const fourParticleRanges = expectCompleteSchedule({
            nSubsteps: 112,
            nParticles: 526_592,
            expectedSubstepsPerSubmission: 8,
        });

        expect(twoParticleRanges).toHaveLength(7);
        expect(fourParticleRanges).toHaveLength(14);
        expect(
            twoParticleRanges[0].endSubstep * 263_296,
        ).toBeLessThanOrEqual(
            MAX_KINEMATIC_COLLIDER_PARTICLE_SUBSTEPS_PER_SUBMISSION,
        );
        expect(
            fourParticleRanges[0].endSubstep * 526_592,
        ).toBeLessThanOrEqual(
            MAX_KINEMATIC_COLLIDER_PARTICLE_SUBSTEPS_PER_SUBMISSION,
        );
    });

    it("submits dense ranges in order with at most four outstanding", async () => {
        const ranges = calculateKinematicColliderSubmissionRanges({
            nSubsteps: 136,
            nParticles: 526_592,
        });
        const submittedRanges: typeof ranges = [];
        const retiredWindowSizes: number[] = [];
        let outstandingSubmissionCount = 0;
        let peakOutstandingSubmissionCount = 0;

        const completedSubsteps =
            await submitKinematicColliderSubmissionRangesInWindows({
                ranges,
                submitRange: range => {
                    submittedRanges.push(range);
                    outstandingSubmissionCount++;
                    peakOutstandingSubmissionCount = Math.max(
                        peakOutstandingSubmissionCount,
                        outstandingSubmissionCount,
                    );
                    return true;
                },
                waitForSubmittedWork: async () => {
                    retiredWindowSizes.push(outstandingSubmissionCount);
                    outstandingSubmissionCount = 0;
                },
            });

        expect(submittedRanges).toEqual(ranges);
        expect(retiredWindowSizes).toEqual([4, 4, 4, 4, 1]);
        expect(peakOutstandingSubmissionCount).toBe(
            MAX_KINEMATIC_COLLIDER_SUBMISSIONS_PER_WINDOW,
        );
        expect(outstandingSubmissionCount).toBe(0);
        expect(completedSubsteps).toBe(136);
    });

    it("returns no submissions for zero work", () => {
        expect(calculateKinematicColliderSubmissionRanges({
            nSubsteps: 0,
            nParticles: 526_592,
        })).toEqual([]);
    });

    it("rejects invalid counts and limits", () => {
        expect(() => calculateKinematicColliderSubmissionRanges({
            nSubsteps: -1,
            nParticles: 1,
        })).toThrow("substep count");
        expect(() => calculateKinematicColliderSubmissionRanges({
            nSubsteps: 1,
            nParticles: 0,
        })).toThrow("particle count");
        expect(() => calculateKinematicColliderSubmissionRanges({
            nSubsteps: 1,
            nParticles: 1,
            maxParticleSubstepsPerSubmission: 0,
        })).toThrow("submission limit");
    });
});
