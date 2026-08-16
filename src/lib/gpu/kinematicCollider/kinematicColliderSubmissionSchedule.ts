// CFL limits one numerical step, not the total GPU work in one command buffer.
// This cap gives the four-particle Zubeia ground eight substeps per submission.
export const MAX_KINEMATIC_COLLIDER_PARTICLE_SUBSTEPS_PER_SUBMISSION =
    4_250_000;
// Four submissions overlap CPU encoding with GPU work while keeping dense
// queued work bounded to 32 Zubeia substeps.
export const MAX_KINEMATIC_COLLIDER_SUBMISSIONS_PER_WINDOW = 4;

export type KinematicColliderSubmissionRange = {
    startSubstep: number,
    endSubstep: number,
};

export type KinematicColliderSubmissionWindowOptions = {
    ranges: readonly KinematicColliderSubmissionRange[],
    submitRange: (range: KinematicColliderSubmissionRange) => boolean,
    waitForSubmittedWork: () => Promise<unknown>,
    maxSubmissionsPerWindow?: number,
};

const assertNonNegativeInteger = (
    value: number,
    label: string,
) => {
    if (!Number.isSafeInteger(value) || value < 0) {
        throw new Error(`${label} must be a non-negative safe integer`);
    }
};

const assertPositiveInteger = (
    value: number,
    label: string,
) => {
    if (!Number.isSafeInteger(value) || value <= 0) {
        throw new Error(`${label} must be a positive safe integer`);
    }
};

export const calculateKinematicColliderSubmissionRanges = ({
    nSubsteps,
    nParticles,
    maxParticleSubstepsPerSubmission =
        MAX_KINEMATIC_COLLIDER_PARTICLE_SUBSTEPS_PER_SUBMISSION,
}: {
    nSubsteps: number,
    nParticles: number,
    maxParticleSubstepsPerSubmission?: number,
}): KinematicColliderSubmissionRange[] => {
    assertNonNegativeInteger(nSubsteps, "substep count");
    assertPositiveInteger(nParticles, "particle count");
    assertPositiveInteger(
        maxParticleSubstepsPerSubmission,
        "particle-substep submission limit",
    );

    const substepsPerSubmission = Math.max(
        1,
        Math.floor(maxParticleSubstepsPerSubmission / nParticles),
    );
    const ranges: KinematicColliderSubmissionRange[] = [];

    for (
        let startSubstep = 0;
        startSubstep < nSubsteps;
        startSubstep += substepsPerSubmission
    ) {
        ranges.push({
            startSubstep,
            endSubstep: Math.min(
                startSubstep + substepsPerSubmission,
                nSubsteps,
            ),
        });
    }

    return ranges;
};

export const submitKinematicColliderSubmissionRangesInWindows = async ({
    ranges,
    submitRange,
    waitForSubmittedWork,
    maxSubmissionsPerWindow =
        MAX_KINEMATIC_COLLIDER_SUBMISSIONS_PER_WINDOW,
}: KinematicColliderSubmissionWindowOptions): Promise<number> => {
    assertPositiveInteger(
        maxSubmissionsPerWindow,
        "submission window size",
    );

    let completedSubsteps = 0;
    for (
        let windowStart = 0;
        windowStart < ranges.length;
        windowStart += maxSubmissionsPerWindow
    ) {
        const windowEnd = Math.min(
            windowStart + maxSubmissionsPerWindow,
            ranges.length,
        );
        let submittedEndSubstep: number | null = null;
        let stopped = false;

        for (
            let rangeIndex = windowStart;
            rangeIndex < windowEnd;
            rangeIndex++
        ) {
            const range = ranges[rangeIndex]!;
            if (!submitRange(range)) {
                stopped = true;
                break;
            }
            submittedEndSubstep = range.endSubstep;
        }

        if (submittedEndSubstep !== null) {
            await waitForSubmittedWork();
            completedSubsteps = submittedEndSubstep;
        }
        if (stopped) break;
    }

    return completedSubsteps;
};
