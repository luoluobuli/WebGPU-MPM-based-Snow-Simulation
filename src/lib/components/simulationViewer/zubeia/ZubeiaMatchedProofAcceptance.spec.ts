import { describe, expect, it } from "vitest";
import {
    ZUBEIA_MATCHED_OFF_CONTACT_P95_MAX_CELLS,
    ZUBEIA_MATCHED_LOCALIZED_ENERGY_MIN_FRACTION,
    ZUBEIA_MATCHED_PAW_CONTACT_DEPRESSION_MIN_CELLS,
    evaluateZubeiaMatchedProofAcceptance,
    type ZubeiaMatchedProofAcceptanceInput,
} from "./ZubeiaMatchedProofAcceptance";

const passingInput = (): ZubeiaMatchedProofAcceptanceInput => ({
    farFieldCenteredVerticalResidualP95M: 0.025,
    pawContactDownwardResidualP95M: 0.0125,
    verticalResidualEnergyInsideDilatedFraction: 0.75,
    surfaceProofMap: {
        cellSizeM: 0.025,
    },
    qualifyingCrackComponents: [{}],
    controlOpeningEdgeCount: 23,
    controlInvalidatedBrokenEdgeCount: 0,
    liftedCrustComponents: [{}],
});

describe("Zubeia matched proof acceptance", () => {
    it("passes every condition at the inclusive thresholds", () => {
        const acceptance = evaluateZubeiaMatchedProofAcceptance(
            passingInput(),
        );

        expect(acceptance).toEqual({
            offContactP95Pass: true,
            pawContactDepressionPass: true,
            localizedEnergyPass: true,
            qualifyingCrackPass: true,
            cleanControlPass: true,
            liftedCrustComponentPass: true,
            footprintPass: true,
            crackPass: true,
            liftedChunkPass: true,
            allPass: true,
        });
        expect(ZUBEIA_MATCHED_OFF_CONTACT_P95_MAX_CELLS).toBe(1);
        expect(ZUBEIA_MATCHED_PAW_CONTACT_DEPRESSION_MIN_CELLS).toBe(0.5);
        expect(ZUBEIA_MATCHED_LOCALIZED_ENERGY_MIN_FRACTION).toBe(0.75);
    });

    it("reports each failed condition and its composite independently", () => {
        const cases: readonly {
            input: ZubeiaMatchedProofAcceptanceInput,
            expected: Record<string, boolean>,
        }[] = [
            {
                input: {
                    ...passingInput(),
                    farFieldCenteredVerticalResidualP95M: 0.025001,
                },
                expected: {
                    offContactP95Pass: false,
                    footprintPass: false,
                    crackPass: true,
                    liftedChunkPass: true,
                },
            },
            {
                input: {
                    ...passingInput(),
                    pawContactDownwardResidualP95M: 0.012499,
                },
                expected: {
                    pawContactDepressionPass: false,
                    footprintPass: false,
                },
            },
            {
                input: {
                    ...passingInput(),
                    verticalResidualEnergyInsideDilatedFraction: 0.749999,
                },
                expected: {
                    localizedEnergyPass: false,
                    footprintPass: false,
                },
            },
            {
                input: {
                    ...passingInput(),
                    qualifyingCrackComponents: [],
                },
                expected: {
                    qualifyingCrackPass: false,
                    crackPass: false,
                },
            },
            {
                input: {
                    ...passingInput(),
                    controlInvalidatedBrokenEdgeCount: 1,
                },
                expected: {
                    cleanControlPass: false,
                    crackPass: false,
                },
            },
            {
                input: {
                    ...passingInput(),
                    liftedCrustComponents: [],
                },
                expected: {
                    liftedCrustComponentPass: false,
                    liftedChunkPass: false,
                },
            },
        ];

        for (const { input, expected } of cases) {
            const acceptance = evaluateZubeiaMatchedProofAcceptance(input);

            expect(acceptance).toMatchObject({
                ...expected,
                allPass: false,
            });
        }
    });

    it("scales displacement thresholds with the grid cell size", () => {
        const input = passingInput();
        input.surfaceProofMap.cellSizeM = 0.05;
        input.farFieldCenteredVerticalResidualP95M = 0.05;
        input.pawContactDownwardResidualP95M = 0.025;

        expect(evaluateZubeiaMatchedProofAcceptance(input)).toMatchObject({
            offContactP95Pass: true,
            pawContactDepressionPass: true,
            footprintPass: true,
        });

        input.farFieldCenteredVerticalResidualP95M = 0.050001;
        expect(evaluateZubeiaMatchedProofAcceptance(input)).toMatchObject({
            offContactP95Pass: false,
            footprintPass: false,
        });
    });

    it("rejects invalid scalar metrics", () => {
        const invalidInputs: readonly [
            ZubeiaMatchedProofAcceptanceInput,
            string,
        ][] = [
            [{
                ...passingInput(),
                surfaceProofMap: { cellSizeM: 0 },
            }, "cell size"],
            [{
                ...passingInput(),
                farFieldCenteredVerticalResidualP95M: Number.NaN,
            }, "off-contact P95"],
            [{
                ...passingInput(),
                pawContactDownwardResidualP95M: -1,
            }, "paw-contact P95"],
            [{
                ...passingInput(),
                verticalResidualEnergyInsideDilatedFraction: 1.01,
            }, "must not exceed one"],
            [{
                ...passingInput(),
                controlOpeningEdgeCount: 0.5,
            }, "control opening edge count"],
            [{
                ...passingInput(),
                controlInvalidatedBrokenEdgeCount: 0.5,
            }, "control-invalidated broken edge count"],
        ];

        for (const [input, message] of invalidInputs) {
            expect(() => evaluateZubeiaMatchedProofAcceptance(input))
                .toThrow(message);
        }
    });
});
