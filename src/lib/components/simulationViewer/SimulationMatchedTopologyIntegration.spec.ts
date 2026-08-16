import {
    beforeEach,
    describe,
    expect,
    it,
    vi,
} from "vitest";

import type {
    RigidPawSdfKinematicColliderAnimation,
} from "$lib/gpu/kinematicCollider/RigidPawSdfKinematicCollider";
import type {
    ZubeiaMatchedTopologyProof,
} from "./zubeia/ZubeiaMatchedTopologyProof";
import type {
    ZubeiaPawContactMask,
} from "./zubeia/ZubeiaPawContactMask";

const mocks = vi.hoisted(() => ({
    analyzeMatchedTopology: vi.fn(),
    buildPawContactMask: vi.fn(),
    createFrameCache: vi.fn(),
}));

vi.mock("./SimulationFrameFileCache", async (importOriginal) => ({
    ...await importOriginal<
        typeof import("./SimulationFrameFileCache")
    >(),
    createSimulationFrameFileCache: mocks.createFrameCache,
}));

vi.mock("./zubeia/ZubeiaMatchedTopologyProof", () => ({
    analyzeZubeiaMatchedCachedFrameTopology:
        mocks.analyzeMatchedTopology,
}));

vi.mock("./zubeia/ZubeiaPawContactMask", () => ({
    buildZubeiaPawContactMask: mocks.buildPawContactMask,
}));

import {
    kinematicGroundFinalScene,
} from "./SimulationScene";
import { SimulationState } from "./SimulationState.svelte";
import {
    ZUBEIA_GROUND_CELL_SIZE_M,
    ZUBEIA_GROUND_STRIP_BOUNDS,
    ZUBEIA_GROUND_SURFACE_Z_M,
} from "./zubeia/ZubeiaGround";


type FrameCache = {
    readFrame: (frameIndex: number) => Promise<ArrayBuffer | null>,
};

type MatchedTopologyInternals = {
    timelineCache: FrameCache | null,
    timelineCacheRootDirectory: FileSystemDirectoryHandle | null,
    counterpartTimelineCacheKey: string,
    retainedTimelineCacheKeys: readonly string[],
    zubeiaRigidPawColliderAnimation:
        RigidPawSdfKinematicColliderAnimation | null,
    updateZubeiaMatchedTopologyProof: (
        frameIndex: number,
        analyzeIntermediate?: boolean,
    ) => Promise<void>,
};

const animation = {
    primitiveCount: 4,
} as RigidPawSdfKinematicColliderAnimation;
const pawContactMask = {
    walkingCellCount: 1,
    crossStripCellCount: 1,
    cellSizeM: ZUBEIA_GROUND_CELL_SIZE_M,
    contact: new Uint8Array(1),
    dilated: new Uint8Array(1),
    perimeter: new Uint8Array(1),
} satisfies ZubeiaPawContactMask;
const matchedProof = {
    particleCount: 1,
} as ZubeiaMatchedTopologyProof;

const configureColliderState = ({
    activeReadFrame,
    rootDirectory,
}: {
    activeReadFrame: FrameCache["readFrame"],
    rootDirectory: FileSystemDirectoryHandle,
}) => {
    const state = new SimulationState({
        scene: kinematicGroundFinalScene,
        pairedTimelineScene: kinematicGroundFinalScene,
        timeline: true,
    });
    const internals = state as unknown as MatchedTopologyInternals;
    internals.timelineCache = {
        readFrame: activeReadFrame,
    };
    internals.timelineCacheRootDirectory = rootDirectory;
    internals.counterpartTimelineCacheKey = "control-key";
    internals.retainedTimelineCacheKeys = [
        "collider-key",
        "control-key",
    ];
    internals.zubeiaRigidPawColliderAnimation = animation;

    return { internals, state };
};


describe("SimulationState matched topology integration", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mocks.buildPawContactMask.mockReturnValue(pawContactMask);
        mocks.analyzeMatchedTopology.mockReturnValue(matchedProof);
    });

    it("reads the matched caches and analyzes the exact paw history", async () => {
        const frameIndex = 12;
        const colliderBaselineFrame = new ArrayBuffer(16);
        const colliderCurrentFrame = new ArrayBuffer(32);
        const controlBaselineFrame = new ArrayBuffer(48);
        const controlCurrentFrame = new ArrayBuffer(64);
        const activeReadFrame = vi.fn(async (index: number) =>
            index === 0 ? colliderBaselineFrame : colliderCurrentFrame
        );
        const controlReadFrame = vi.fn(async (index: number) =>
            index === 0 ? controlBaselineFrame : controlCurrentFrame
        );
        const rootDirectory = {
            name: "paired-cache",
        } as FileSystemDirectoryHandle;
        mocks.createFrameCache.mockResolvedValue({
            readFrame: controlReadFrame,
        });
        const { internals, state } = configureColliderState({
            activeReadFrame,
            rootDirectory,
        });

        await internals.updateZubeiaMatchedTopologyProof(
            frameIndex,
            true,
        );

        expect(activeReadFrame.mock.calls).toEqual([[0], [frameIndex]]);
        expect(mocks.createFrameCache).toHaveBeenCalledWith({
            cacheKey: "control-key",
            rootDirectory,
            retainedCacheKeys: ["collider-key", "control-key"],
        });
        expect(controlReadFrame.mock.calls).toEqual([[0], [frameIndex]]);
        expect(mocks.buildPawContactMask).toHaveBeenCalledWith({
            animation,
            groundStripBounds: ZUBEIA_GROUND_STRIP_BOUNDS,
            cellSizeM: ZUBEIA_GROUND_CELL_SIZE_M,
            groundSurfaceZM: ZUBEIA_GROUND_SURFACE_Z_M,
            outputFrameIndex: frameIndex,
            secondsPerFrame: 1 / 30,
        });
        expect(mocks.analyzeMatchedTopology).toHaveBeenCalledWith({
            colliderBaselineFrame,
            colliderCurrentFrame,
            controlBaselineFrame,
            controlCurrentFrame,
            gridCellSizeM: ZUBEIA_GROUND_CELL_SIZE_M,
            structuralGroundMask: pawContactMask,
        });
        expect(state.zubeiaPawContactMask).toBe(pawContactMask);
        expect(state.zubeiaTopologyProof).toBe(matchedProof);
        expect(state.zubeiaTopologyProofStatus).toContain(
            `through frame ${frameIndex}`,
        );
    });

    it("fails closed when the matched control is not baked", async () => {
        const frameIndex = 20;
        const activeReadFrame = vi.fn(async () => new ArrayBuffer(16));
        const controlReadFrame = vi.fn(async (index: number) =>
            index === 0 ? new ArrayBuffer(16) : null
        );
        mocks.createFrameCache.mockResolvedValue({
            readFrame: controlReadFrame,
        });
        const { internals, state } = configureColliderState({
            activeReadFrame,
            rootDirectory: {
                name: "paired-cache",
            } as FileSystemDirectoryHandle,
        });

        await expect(
            internals.updateZubeiaMatchedTopologyProof(frameIndex, true),
        ).resolves.toBeUndefined();

        expect(state.zubeiaTopologyProof).toBeNull();
        expect(state.zubeiaPawContactMask).toBeNull();
        expect(state.zubeiaTopologyProofStatus).toBe(
            `Bake matched control through frame ${frameIndex}`,
        );
        expect(mocks.buildPawContactMask).not.toHaveBeenCalled();
        expect(mocks.analyzeMatchedTopology).not.toHaveBeenCalled();
    });

    it("identifies the control without attempting proof", async () => {
        const state = new SimulationState({
            scene: {
                ...kinematicGroundFinalScene,
                colliderSource: null,
                renderCollider: false,
            },
            pairedTimelineScene: kinematicGroundFinalScene,
            timeline: true,
        });
        const internals = state as unknown as MatchedTopologyInternals;

        await internals.updateZubeiaMatchedTopologyProof(12, true);

        expect(state.zubeiaTopologyProof).toBeNull();
        expect(state.zubeiaPawContactMask).toBeNull();
        expect(state.zubeiaTopologyProofStatus).toContain(
            "Matched control bake",
        );
        expect(state.zubeiaTopologyProofStatus).toContain(
            "proof is shown on the collider route",
        );
        expect(mocks.createFrameCache).not.toHaveBeenCalled();
        expect(mocks.buildPawContactMask).not.toHaveBeenCalled();
        expect(mocks.analyzeMatchedTopology).not.toHaveBeenCalled();
    });

    it("keeps automatic analysis gated to the final frame", async () => {
        const activeReadFrame = vi.fn(async () => new ArrayBuffer(16));
        const { internals } = configureColliderState({
            activeReadFrame,
            rootDirectory: {
                name: "paired-cache",
            } as FileSystemDirectoryHandle,
        });

        await internals.updateZubeiaMatchedTopologyProof(12);

        expect(activeReadFrame).not.toHaveBeenCalled();
        expect(mocks.createFrameCache).not.toHaveBeenCalled();
        expect(mocks.analyzeMatchedTopology).not.toHaveBeenCalled();
    });
});