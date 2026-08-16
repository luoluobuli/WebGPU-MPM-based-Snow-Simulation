import { describe, expect, it } from "vitest";

import viewerSrc from "./SimulationViewer.svelte?raw";

describe("SimulationViewer ground proof routes", () => {
    it("requires the kinematic timeline for inspection", () => {
        expect(viewerSrc).toContain(
            'const inspect = kinematic && urlSearchParams?.has("inspect") === true',
        );
        expect(viewerSrc).toContain(
            'urlSearchParams?.has("collider") === true',
        );
        expect(viewerSrc).toContain("renderCollider: inspectCollider");
        expect(viewerSrc).toContain("radius: 2.7");
        expect(viewerSrc).toContain("lat: Math.PI * 0.4");
        expect(viewerSrc).toContain("long: Math.PI * 0.5");
        expect(viewerSrc).toContain("offset: [0, 0.9, 0.05]");
    });

    it("provides a matched dense control with only the collider removed", () => {
        expect(viewerSrc).toContain(
            'final && urlSearchParams?.has("control") === true',
        );
        expect(viewerSrc).toContain("colliderSource: null");
        expect(viewerSrc).toContain("renderCollider: false");
        expect(viewerSrc).toContain(
            "getPairedTimelineScene: () =>",
        );
    });
});