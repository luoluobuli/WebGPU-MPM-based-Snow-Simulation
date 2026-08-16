export type ZubeiaMatchedProofAcceptanceInput = {
    farFieldCenteredVerticalResidualP95M: number,
    pawContactDownwardResidualP95M: number,
    verticalResidualEnergyInsideDilatedFraction: number,
    surfaceProofMap: {
        cellSizeM: number,
    },
    qualifyingCrackComponents: readonly unknown[],
    controlOpeningEdgeCount: number,
    controlInvalidatedBrokenEdgeCount: number,
    liftedCrustComponents: readonly unknown[],
};

export type ZubeiaMatchedProofAcceptance = {
    offContactP95Pass: boolean,
    pawContactDepressionPass: boolean,
    localizedEnergyPass: boolean,
    qualifyingCrackPass: boolean,
    cleanControlPass: boolean,
    liftedCrustComponentPass: boolean,
    footprintPass: boolean,
    crackPass: boolean,
    liftedChunkPass: boolean,
    allPass: boolean,
};

export const ZUBEIA_MATCHED_OFF_CONTACT_P95_MAX_CELLS = 1;
export const ZUBEIA_MATCHED_PAW_CONTACT_DEPRESSION_MIN_CELLS = 0.5;
export const ZUBEIA_MATCHED_LOCALIZED_ENERGY_MIN_FRACTION = 0.75;

const assertFiniteNonnegative = (value: number, label: string) => {
    if (Number.isFinite(value) && value >= 0) return;

    throw new Error(`Zubeia matched ${label} must be finite and nonnegative`);
};

export const evaluateZubeiaMatchedProofAcceptance = (
    proof: ZubeiaMatchedProofAcceptanceInput,
): ZubeiaMatchedProofAcceptance => {
    const cellSizeM = proof.surfaceProofMap.cellSizeM;
    if (!Number.isFinite(cellSizeM) || cellSizeM <= 0) {
        throw new Error("Zubeia matched proof cell size must be finite and positive");
    }
    assertFiniteNonnegative(
        proof.farFieldCenteredVerticalResidualP95M,
        "off-contact P95",
    );
    assertFiniteNonnegative(
        proof.pawContactDownwardResidualP95M,
        "paw-contact P95",
    );
    assertFiniteNonnegative(
        proof.verticalResidualEnergyInsideDilatedFraction,
        "localized energy fraction",
    );
    if (proof.verticalResidualEnergyInsideDilatedFraction > 1) {
        throw new Error(
            "Zubeia matched localized energy fraction must not exceed one",
        );
    }
    if (
        !Number.isInteger(proof.controlOpeningEdgeCount)
        || proof.controlOpeningEdgeCount < 0
    ) {
        throw new Error(
            "Zubeia matched control opening edge count must be a nonnegative integer",
        );
    }
    if (
        !Number.isInteger(proof.controlInvalidatedBrokenEdgeCount)
        || proof.controlInvalidatedBrokenEdgeCount < 0
    ) {
        throw new Error(
            "Zubeia matched control-invalidated broken edge count must be a nonnegative integer",
        );
    }

    const offContactP95Pass =
        proof.farFieldCenteredVerticalResidualP95M
        <= ZUBEIA_MATCHED_OFF_CONTACT_P95_MAX_CELLS * cellSizeM;
    const pawContactDepressionPass =
        proof.pawContactDownwardResidualP95M
        >= ZUBEIA_MATCHED_PAW_CONTACT_DEPRESSION_MIN_CELLS * cellSizeM;
    const localizedEnergyPass =
        proof.verticalResidualEnergyInsideDilatedFraction
        >= ZUBEIA_MATCHED_LOCALIZED_ENERGY_MIN_FRACTION;
    const qualifyingCrackPass = proof.qualifyingCrackComponents.length > 0;
    const cleanControlPass = proof.controlInvalidatedBrokenEdgeCount === 0;
    const liftedCrustComponentPass = proof.liftedCrustComponents.length > 0;
    const footprintPass =
        offContactP95Pass
        && pawContactDepressionPass
        && localizedEnergyPass;
    const crackPass = qualifyingCrackPass && cleanControlPass;
    const liftedChunkPass = liftedCrustComponentPass;

    return {
        offContactP95Pass,
        pawContactDepressionPass,
        localizedEnergyPass,
        qualifyingCrackPass,
        cleanControlPass,
        liftedCrustComponentPass,
        footprintPass,
        crackPass,
        liftedChunkPass,
        allPass: footprintPass && crackPass && liftedChunkPass,
    };
};
