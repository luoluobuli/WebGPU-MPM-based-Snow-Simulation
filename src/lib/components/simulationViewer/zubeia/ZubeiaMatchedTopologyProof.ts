import type { ZubeiaPawContactMask } from "./ZubeiaPawContactMask";
import { ZUBEIA_GROUND_CRUST_MATERIAL_ID } from "./ZubeiaGround";
import { buildMatchedSurfaceProof } from "./ZubeiaMatchedSurfaceProof";
import {
    ZUBEIA_MATCHED_CRACK_MIN_OPENING_CELLS,
    ZUBEIA_MATCHED_LIFT_MIN_HEIGHT_CELLS,
    analyzeMatchedCracks,
    buildMatchedLiftedCrustComponents,
} from "./ZubeiaMatchedTopologyComponents";
import {
    componentAt,
    buildBaselineTopCrustParticleMask,
    maskIndexForPosition,
    materialAt,
    sortedMedian,
    sortedP95,
    validateMatchedFrames,
    validateMatchedMask,
} from "./ZubeiaMatchedTopologyFrame";

type Vec2 = readonly [number, number];
type Vec3 = readonly [number, number, number];

export type ZubeiaStructuralGroundMask = ZubeiaPawContactMask;

export type ZubeiaMatchedSurfaceProofMap = {
    walkingCellCount: number,
    crossStripCellCount: number,
    cellSizeM: number,
    colliderSurfacePresent: Uint8Array,
    controlSurfacePresent: Uint8Array,
    matchedSurfacePresent: Uint8Array,
    colliderTopZM: Float32Array,
    controlTopZM: Float32Array,
    matchedDepressionM: Float32Array,
};

export type ZubeiaMatchedLiftedCrustComponent = {
    particleCount: number,
    particleIndices: readonly number[],
    centroidM: Vec3,
    centroidLiftM: number,
    maxLiftM: number,
    boundsMinM: Vec3,
    boundsMaxM: Vec3,
};

export type ZubeiaMatchedCrackComponent = {
    edgeCount: number,
    particleCount: number,
    particleIndices: readonly number[],
    cellIndices: readonly number[],
    meanMatchedOpeningM: number,
    maxMatchedOpeningM: number,
    boundsMinM: Vec2,
    boundsMaxM: Vec2,
    intersectsPawPerimeter: boolean,
};

export type ZubeiaMatchedTopologyProof = {
    particleCount: number,
    crustParticleCount: number,
    farFieldVerticalResidualCenterM: number,
    farFieldCenteredVerticalResidualP95M: number,
    pawContactDownwardResidualP95M: number,
    verticalResidualEnergyInsideDilatedFraction: number,
    maxMatchedSurfaceDepressionM: number,
    surfaceProofMap: ZubeiaMatchedSurfaceProofMap,
    liftedCrustComponents: ZubeiaMatchedLiftedCrustComponent[],
    matchedBrokenEdgeCount: number,
    controlOpeningEdgeCount: number,
    controlInvalidatedBrokenEdgeCount: number,
    crackComponents: ZubeiaMatchedCrackComponent[],
    qualifyingCrackComponents: ZubeiaMatchedCrackComponent[],
    crackCellMask: Uint8Array,
    controlOpeningCellMask: Uint8Array,
};

export {
    ZUBEIA_MATCHED_CRACK_MIN_OPENING_CELLS,
    ZUBEIA_MATCHED_LIFT_MIN_HEIGHT_CELLS,
};

const calculateResidualMetrics = ({
    colliderBaseline,
    colliderCurrent,
    controlBaseline,
    controlCurrent,
    particleCount,
    topCrustParticleMask,
    mask,
}: {
    colliderBaseline: ReturnType<typeof validateMatchedFrames>["views"][0],
    colliderCurrent: ReturnType<typeof validateMatchedFrames>["views"][0],
    controlBaseline: ReturnType<typeof validateMatchedFrames>["views"][0],
    controlCurrent: ReturnType<typeof validateMatchedFrames>["views"][0],
    particleCount: number,
    topCrustParticleMask: Uint8Array,
    mask: ZubeiaPawContactMask,
}) => {
    const residualZ = new Float32Array(particleCount);
    const particleMaskIndices = new Int32Array(particleCount);
    const samples = new Float64Array(particleCount);
    let crustParticleCount = 0;
    let farFieldCount = 0;

    for (let particleIndex = 0; particleIndex < particleCount; particleIndex++) {
        if (
            materialAt(colliderBaseline, particleIndex)
            === ZUBEIA_GROUND_CRUST_MATERIAL_ID
        ) {
            crustParticleCount++;
        }

        const residual =
            componentAt(colliderCurrent, particleIndex, 2)
            - componentAt(colliderBaseline, particleIndex, 2)
            - componentAt(controlCurrent, particleIndex, 2)
            + componentAt(controlBaseline, particleIndex, 2);
        residualZ[particleIndex] = residual;
        if (topCrustParticleMask[particleIndex] === 0) continue;

        const maskIndex = maskIndexForPosition(
            componentAt(colliderBaseline, particleIndex, 0),
            componentAt(colliderBaseline, particleIndex, 1),
            mask,
        );
        particleMaskIndices[particleIndex] = maskIndex;
        if (maskIndex < 0 || mask.dilated[maskIndex] === 0) {
            samples[farFieldCount++] = residual;
        }
    }

    const residualCenterM = sortedMedian(samples, farFieldCount);
    let totalEnergy = 0;
    let insideDilatedEnergy = 0;
    let centeredFarFieldCount = 0;
    let contactCount = 0;
    for (let particleIndex = 0; particleIndex < particleCount; particleIndex++) {
        if (topCrustParticleMask[particleIndex] === 0) continue;

        const centeredResidualM = residualZ[particleIndex] - residualCenterM;
        const maskIndex = particleMaskIndices[particleIndex];
        const energy = centeredResidualM * centeredResidualM;
        totalEnergy += energy;

        if (maskIndex >= 0 && mask.dilated[maskIndex] !== 0) {
            insideDilatedEnergy += energy;
        } else {
            samples[centeredFarFieldCount++] = Math.abs(centeredResidualM);
        }
    }
    const farFieldP95M = sortedP95(samples, centeredFarFieldCount);

    for (let particleIndex = 0; particleIndex < particleCount; particleIndex++) {
        if (topCrustParticleMask[particleIndex] === 0) continue;

        const maskIndex = particleMaskIndices[particleIndex];
        if (maskIndex < 0 || mask.contact[maskIndex] === 0) continue;

        samples[contactCount++] = Math.max(
            0,
            -(residualZ[particleIndex] - residualCenterM),
        );
    }

    return {
        crustParticleCount,
        residualZ,
        farFieldVerticalResidualCenterM: residualCenterM,
        farFieldCenteredVerticalResidualP95M: farFieldP95M,
        pawContactDownwardResidualP95M: sortedP95(samples, contactCount),
        verticalResidualEnergyInsideDilatedFraction:
            totalEnergy > 0 ? insideDilatedEnergy / totalEnergy : 0,
    };
};

export const analyzeZubeiaMatchedCachedFrameTopology = ({
    colliderBaselineFrame,
    colliderCurrentFrame,
    controlBaselineFrame,
    controlCurrentFrame,
    gridCellSizeM,
    structuralGroundMask,
}: {
    colliderBaselineFrame: ArrayBuffer,
    colliderCurrentFrame: ArrayBuffer,
    controlBaselineFrame: ArrayBuffer,
    controlCurrentFrame: ArrayBuffer,
    gridCellSizeM: number,
    structuralGroundMask: ZubeiaStructuralGroundMask,
}): ZubeiaMatchedTopologyProof => {
    if (!Number.isFinite(gridCellSizeM) || gridCellSizeM <= 0) {
        throw new Error("Zubeia matched topology grid cell size must be positive");
    }
    validateMatchedMask(structuralGroundMask, gridCellSizeM);

    const {
        views: [
            colliderBaseline,
            colliderCurrent,
            controlBaseline,
            controlCurrent,
        ],
        particleCount,
    } = validateMatchedFrames([
        colliderBaselineFrame,
        colliderCurrentFrame,
        controlBaselineFrame,
        controlCurrentFrame,
    ]);
    const topCrustParticleMask = buildBaselineTopCrustParticleMask({
        baseline: colliderBaseline,
        particleCount,
        mask: structuralGroundMask,
        gridCellSizeM,
    });
    const residual = calculateResidualMetrics({
        colliderBaseline,
        colliderCurrent,
        controlBaseline,
        controlCurrent,
        particleCount,
        topCrustParticleMask,
        mask: structuralGroundMask,
    });
    const surface = buildMatchedSurfaceProof({
        colliderCurrent,
        controlCurrent,
        particleCount,
        mask: structuralGroundMask,
    });
    const liftedCrustComponents = buildMatchedLiftedCrustComponents({
        colliderBaseline,
        colliderCurrent,
        particleCount,
        residualZ: residual.residualZ,
        residualCenterM: residual.farFieldVerticalResidualCenterM,
        gridCellSizeM,
    });
    const cracks = analyzeMatchedCracks({
        colliderBaseline,
        colliderCurrent,
        controlCurrent,
        particleCount,
        topCrustParticleMask,
        mask: structuralGroundMask,
        gridCellSizeM,
    });

    return {
        particleCount,
        crustParticleCount: residual.crustParticleCount,
        farFieldVerticalResidualCenterM:
            residual.farFieldVerticalResidualCenterM,
        farFieldCenteredVerticalResidualP95M:
            residual.farFieldCenteredVerticalResidualP95M,
        pawContactDownwardResidualP95M:
            residual.pawContactDownwardResidualP95M,
        verticalResidualEnergyInsideDilatedFraction:
            residual.verticalResidualEnergyInsideDilatedFraction,
        maxMatchedSurfaceDepressionM: surface.maxMatchedSurfaceDepressionM,
        surfaceProofMap: surface.surfaceProofMap,
        liftedCrustComponents,
        ...cracks,
    };
};
