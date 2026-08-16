export const KEYED_SDF_VALIDATION_THRESHOLDS = {
    minimumTriangleDoubleAreaM2: 1e-10,
    minimumClosedComponentVolumeM3: 1e-9,
    capHeightToleranceM: 1e-6,
    minimumNearSurfaceGradientLength: 0.25,
    minimumSoleNormalDot: 0.5,
    maximumSurfaceVelocityErrorMPerS: 0.35,
    minimumSignedProbeMagnitudeM: 0.001,
    minimumSignedProbesPerPawSample: 2,
    maximumSignedProbesPerPawSample: 16,
    maximumSoleOutwardNormalZ: -0.25,
    signedProbeOffsetCellFractions: [
        1.25,
        0.875,
        0.625,
    ],
    surfaceBarycentricSamples: [
        [1 / 3, 1 / 3, 1 / 3],
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6],
    ],
} as const;
