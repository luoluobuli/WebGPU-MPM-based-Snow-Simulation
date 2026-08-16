import { describe, expect, it } from "vitest";

import {
    kinematicGroundFinalScene,
} from "./SimulationScene";
import { SimulationState } from "./SimulationState.svelte";


type PairedTimelineInternals = {
    pairedTimelineColliderSimulationContentFingerprint: string,
    pairedTimelineControlSimulationContentFingerprint: string,
    counterpartTimelineCacheKey: string,
    retainedTimelineCacheKeys: readonly string[],
    currentTimelineCacheKey: () => string,
};

const configureContentFingerprints = (
    state: SimulationState,
) => {
    const internals = state as unknown as PairedTimelineInternals;
    internals.pairedTimelineColliderSimulationContentFingerprint =
        "spawn-and-zubeia";
    internals.pairedTimelineControlSimulationContentFingerprint =
        "spawn-and-null";

    return internals;
};


describe("SimulationState paired timeline cache keys", () => {
    it("cross-matches the actual page keys and retains both", () => {
        const colliderInternals = configureContentFingerprints(
            new SimulationState({
                scene: kinematicGroundFinalScene,
                pairedTimelineScene: kinematicGroundFinalScene,
                timeline: true,
            }),
        );
        const controlInternals = configureContentFingerprints(
            new SimulationState({
                scene: {
                    ...kinematicGroundFinalScene,
                    colliderSource: null,
                    renderCollider: false,
                },
                pairedTimelineScene: kinematicGroundFinalScene,
                timeline: true,
            }),
        );

        const colliderPageKey =
            colliderInternals.currentTimelineCacheKey();
        const controlPageKey =
            controlInternals.currentTimelineCacheKey();

        expect(colliderPageKey).toBe(
            controlInternals.counterpartTimelineCacheKey,
        );
        expect(controlPageKey).toBe(
            colliderInternals.counterpartTimelineCacheKey,
        );
        expect(colliderInternals.retainedTimelineCacheKeys).toEqual([
            colliderPageKey,
            controlPageKey,
        ]);
        expect(controlInternals.retainedTimelineCacheKeys).toEqual([
            controlPageKey,
            colliderPageKey,
        ]);
    });
});
