<script module lang="ts">
export const ZUBEIA_QUALIFYING_CRACK_COMPONENT_DISPLAY_LIMIT = 8;
</script>

<script lang="ts">
import ElapsedTimeDisplay from "./ElapsedTimeDisplay.svelte";
import OverlayPanel from "./OverlayPanel.svelte";
import Separator from "$lib/components/generic/Separator.svelte";
import type { SimulationState } from "./SimulationState.svelte";
import ZubeiaSurfaceProofMap from "./zubeia/ZubeiaSurfaceProofMap.svelte";
import {
    ZUBEIA_MATCHED_LOCALIZED_ENERGY_MIN_FRACTION,
    ZUBEIA_MATCHED_OFF_CONTACT_P95_MAX_CELLS,
    ZUBEIA_MATCHED_PAW_CONTACT_DEPRESSION_MIN_CELLS,
    evaluateZubeiaMatchedProofAcceptance,
} from "./zubeia/ZubeiaMatchedProofAcceptance";

let {
    simulationState,
    status,
    err,
}: {
    simulationState: SimulationState,
    status: string,
    err: string | null,
} = $props();

const nPrerenderPasses = $derived(simulationState.prerenderElapsedTimes?.length);
const zubeiaTopologyProof = $derived(simulationState.zubeiaTopologyProof);
const zubeiaPawContactMask = $derived(simulationState.zubeiaPawContactMask);
const zubeiaMatchedAcceptance = $derived(
    zubeiaTopologyProof === null
        ? null
        : evaluateZubeiaMatchedProofAcceptance(zubeiaTopologyProof),
);
const displayedQualifyingCrackComponents = $derived(
    (zubeiaTopologyProof?.qualifyingCrackComponents ?? [])
        .slice(0, ZUBEIA_QUALIFYING_CRACK_COMPONENT_DISPLAY_LIMIT),
);
const remainingQualifyingCrackComponentCount = $derived(
    (zubeiaTopologyProof?.qualifyingCrackComponents.length ?? 0)
    - displayedQualifyingCrackComponents.length,
);
const zubeiaGridCellSizeMm = $derived(
    (zubeiaTopologyProof?.surfaceProofMap.cellSizeM ?? 0) * 1000,
);
const zubeiaOffContactP95MaxMm = $derived(
    ZUBEIA_MATCHED_OFF_CONTACT_P95_MAX_CELLS * zubeiaGridCellSizeMm,
);
const zubeiaPawContactP95MinMm = $derived(
    ZUBEIA_MATCHED_PAW_CONTACT_DEPRESSION_MIN_CELLS * zubeiaGridCellSizeMm,
);
const acceptanceLabel = (passed: boolean) => passed ? "PASS" : "FAIL";
</script>

<OverlayPanel>
    <h3>Status</h3>
    <div>{err !== null ? `error: ${err}` : status}</div>

    {#if simulationState.zubeiaTopologyProofStatus !== null || zubeiaTopologyProof !== null}
        <Separator />

        <h3>Matched ground proof</h3>
        {#if simulationState.zubeiaTopologyProofStatus !== null}
            <div>{simulationState.zubeiaTopologyProofStatus}</div>
        {/if}

        {#if zubeiaTopologyProof !== null
            && zubeiaPawContactMask !== null
            && zubeiaMatchedAcceptance !== null}
            <div>
                {(zubeiaTopologyProof.maxMatchedSurfaceDepressionM * 1000).toFixed(1)}
                mm maximum matched surface depression
            </div>
            <div>
                {(zubeiaTopologyProof.farFieldCenteredVerticalResidualP95M * 1000).toFixed(1)}
                mm off-contact residual P95; maximum one grid cell
                ({zubeiaOffContactP95MaxMm.toFixed(1)} mm):
                {acceptanceLabel(zubeiaMatchedAcceptance.offContactP95Pass)}
            </div>
            <div>
                {(zubeiaTopologyProof.pawContactDownwardResidualP95M * 1000).toFixed(1)}
                mm paw-contact downward residual P95; minimum half grid cell
                ({zubeiaPawContactP95MinMm.toFixed(1)} mm):
                {acceptanceLabel(zubeiaMatchedAcceptance.pawContactDepressionPass)}
            </div>
            <div>
                {(zubeiaTopologyProof.verticalResidualEnergyInsideDilatedFraction * 100).toFixed(1)}%
                residual energy near paw contact; minimum
                {(ZUBEIA_MATCHED_LOCALIZED_ENERGY_MIN_FRACTION * 100).toFixed(1)}%:
                {acceptanceLabel(zubeiaMatchedAcceptance.localizedEnergyPass)}
            </div>

            <ZubeiaSurfaceProofMap
                proof={zubeiaTopologyProof}
                pawContactMask={zubeiaPawContactMask}
            />

            <div>
                {zubeiaTopologyProof.crackComponents.length} matched crack components;
                {zubeiaTopologyProof.qualifyingCrackComponents.length} qualifying:
                {acceptanceLabel(zubeiaMatchedAcceptance.qualifyingCrackPass)}
            </div>
            <div>
                {zubeiaTopologyProof.matchedBrokenEdgeCount} matched broken edges;
                {zubeiaTopologyProof.controlOpeningEdgeCount} total control openings
                (reported only)
            </div>
            <div>
                {zubeiaTopologyProof.controlInvalidatedBrokenEdgeCount}
                matched broken edges invalidated by control; clean control requires zero:
                {acceptanceLabel(zubeiaMatchedAcceptance.cleanControlPass)}
            </div>
            {#each displayedQualifyingCrackComponents as crack, crackIndex}
                <div>
                    crack {crackIndex + 1}: {crack.edgeCount} edges;
                    {(crack.maxMatchedOpeningM * 1000).toFixed(1)} mm maximum opening;
                    {crack.intersectsPawPerimeter ? "at" : "away from"} paw perimeter
                </div>
            {/each}
            {#if remainingQualifyingCrackComponentCount > 0}
                <div>
                    {remainingQualifyingCrackComponentCount} additional qualifying crack components not shown
                </div>
            {/if}
            <div>
                {zubeiaTopologyProof.liftedCrustComponents.length} lifted crust groups:
                {acceptanceLabel(zubeiaMatchedAcceptance.liftedCrustComponentPass)}
            </div>
            <div>
                footprint {acceptanceLabel(zubeiaMatchedAcceptance.footprintPass)};
                cracks {acceptanceLabel(zubeiaMatchedAcceptance.crackPass)};
                lifted chunks {acceptanceLabel(zubeiaMatchedAcceptance.liftedChunkPass)};
                overall {acceptanceLabel(zubeiaMatchedAcceptance.allPass)}
            </div>
        {/if}
    {/if}
    <Separator />

    <h3>Performance</h3>

    <h4>Simulation</h4>

    <perf-item>
        <perf-label>GPU simulation compute pass ({simulationState.elapsedTime.nSimulationSubsteps} substeps)</perf-label>
        <ElapsedTimeDisplay
            ns={simulationState.elapsedTime.gpuComputeSimulationStepTimeNs}
        />
    </perf-item>

    <perf-item>
        <perf-label>GPU simulation substep (sample)</perf-label>
        <ElapsedTimeDisplay
            ns={simulationState.elapsedTime.gpuComputeSimulationSubstepTimeNs}
        />
    </perf-item>

    <h4>Render</h4>

    <div>{nPrerenderPasses ?? "(unknown)"} prerender {nPrerenderPasses === 1 ? "pass" : "passes"}</div>

    {#each simulationState.prerenderElapsedTimes as prerenderElapsedTime (prerenderElapsedTime.label)}
    <perf-item>
        <perf-label>GPU {prerenderElapsedTime.label} pass (sample)</perf-label>
        <ElapsedTimeDisplay
            ns={prerenderElapsedTime.elapsedTimeNs ?? 0n}
        />
    </perf-item>
    {/each}

    <perf-item>
        <perf-label>GPU composite render pass (sample)</perf-label>
        <ElapsedTimeDisplay
            ns={simulationState.elapsedTime.gpuRenderTimeNs}
        />
    </perf-item>

    <perf-item>
        <perf-label>GPU postprocess render pass (sample)</perf-label>
        <ElapsedTimeDisplay
            ns={simulationState.elapsedTime.gpuPostprocessRenderTimeNs}
        />
    </perf-item>

    <h4>Overall</h4>
    <perf-item>
        <perf-label>Total animation frame time</perf-label>
        <ElapsedTimeDisplay
            ns={simulationState.elapsedTime.animationFrameTimeNs}
            showMsFractionalPart={false}
            inverseLabel="fps"
        />
    </perf-item>
</OverlayPanel>
