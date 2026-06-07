<script lang="ts">
import { EntrySlider, Hotkey } from "@vaie/hui";
import type { SimulationState } from "./SimulationState.svelte";
import Separator from "$lib/components/generic/Separator.svelte";
import Button from "$lib/components/generic/Button.svelte";
import { GpuRenderMethodType } from "$lib/gpu/GpuRenderMethod";
import { GpuSimulationMethodType } from "$lib/gpu/GpuSimulationMethod";
import OverlayPanel from "./OverlayPanel.svelte";
import { ParticleControlMode } from "./ParticleControlMode";

let {
    simulationState,
}: {
    simulationState: SimulationState,
} = $props();


const MIN_TIMESTEP_DIVISOR = 15;
const MAX_TIMESTEP_DIVISOR = 10_000;


const progressFromTimestep = (timestep: number) => {
    return 1 - Math.log((1 / timestep) / MIN_TIMESTEP_DIVISOR) / Math.log(MAX_TIMESTEP_DIVISOR / MIN_TIMESTEP_DIVISOR);
};
const formatFixed3 = (value: number) => value.toFixed(3);
const formatTimestepDivisor = (value: number) => `1 / ${value.toFixed(1)} s`;
const parseTimestepDivisor = (value: string) => {
    const trimmed = value.trim();
    const reciprocalMatch = /^1\s*\/\s*(\d+(?:\.\d+)?)\s*s?$/i.exec(trimmed);
    const numericText = reciprocalMatch?.[1] ?? trimmed.replace(/\s*s$/i, "");
    const parsed = Number(numericText);

    return Number.isFinite(parsed) ? parsed : null;
};

const timestepProgress = $derived.by(() => {
    switch (simulationState.simulationMethodType) {
        case GpuSimulationMethodType.ExplicitMpm:
            return progressFromTimestep(simulationState.explicitMpmMaxSimulationTimestepS);

        case GpuSimulationMethodType.MlsMpm:
        case GpuSimulationMethodType.FusedMlsMpm:
            return progressFromTimestep(simulationState.mlsMpmMaxSimulationTimestepS);
    }
});

const timestepDivisor = $derived(Math.pow(MAX_TIMESTEP_DIVISOR / MIN_TIMESTEP_DIVISOR, 1 - timestepProgress) * MIN_TIMESTEP_DIVISOR);
const actualTimestepDivisor = $derived(simulationState.actualSimulationTimestepS === null
    ? null
    : 1 / simulationState.actualSimulationTimestepS);
const isTimestepCflLimited = $derived(actualTimestepDivisor !== null && actualTimestepDivisor > timestepDivisor * 1.001);

const updateTimestepDivisor = (divisor: number) => {
    switch (simulationState.simulationMethodType) {
        case GpuSimulationMethodType.ExplicitMpm:
            simulationState.explicitMpmMaxSimulationTimestepS = 1 / divisor;
            break;

        case GpuSimulationMethodType.MlsMpm:
        case GpuSimulationMethodType.FusedMlsMpm:
            simulationState.mlsMpmMaxSimulationTimestepS = 1 / divisor;
            break;
    }
};
</script>

<OverlayPanel>
    <h3>Render method</h3>

    <label>
        <input
            type="radio"
            name="render-method-type"
            bind:group={simulationState.renderMethodType}
            value={GpuRenderMethodType.Points}
        />
        Points
    </label>

    <label>
        <input
            type="radio"
            name="render-method-type"
            bind:group={simulationState.renderMethodType}
            value={GpuRenderMethodType.Splats}
        />
        Splats
    </label>

    <label>
        <input
            type="radio"
            name="render-method-type"
            bind:group={simulationState.renderMethodType}
            value={GpuRenderMethodType.Volumetric}
        />
        Volumetric
    </label>

    <label>
        <input
            type="radio"
            name="render-method-type"
            bind:group={simulationState.renderMethodType}
            value={GpuRenderMethodType.Ssfr}
        />
        SSFR
    </label>

    <label>
        <input
            type="radio"
            name="render-method-type"
            bind:group={simulationState.renderMethodType}
            value={GpuRenderMethodType.MarchingCubes}
        />
        Marching cubes
    </label>

    <label>
        <input
            type="radio"
            name="render-method-type"
            bind:group={simulationState.renderMethodType}
            value={GpuRenderMethodType.RaymarchingSurface}
        />
        Raymarching surface
    </label>

    <Separator />

    <h3>Simulation</h3>

    <div>
        <Hotkey
            key="r"
            onKeyUp={() => simulationState.restart()}
        >
            {#snippet pressTarget({keyHeld})}
                <Button
                    {keyHeld}
                    onclick={() => simulationState.restart()}
                >Restart (R)</Button>
            {/snippet}
        </Hotkey>
    </div>

    <h4>Control</h4>

    <div>Right-click particles to:</div>

    <label>
        <input
            type="radio"
            name="particle-control-mode"
            bind:group={simulationState.particleControlMode}
            value={ParticleControlMode.Repel}
        />
        Repel
    </label>

    <label>
        <input
            type="radio"
            name="particle-control-mode"
            bind:group={simulationState.particleControlMode}
            value={ParticleControlMode.Attract}
        />
        Attract
    </label>

    <div>Interaction radius</div>

    <number-control>
        <EntrySlider
            value={simulationState.interactionRadiusFactor}
            onValueChange={(value) => simulationState.interactionRadiusFactor = value}
            min={0}
            max={15}
            softMin={0}
            softMax={15}
            step={0.001}
            format={formatFixed3}
        />
    </number-control>

    <div>Interaction strength</div>

    <number-control>
        <EntrySlider
            value={simulationState.interactionStrength}
            onValueChange={(value) => simulationState.interactionStrength = value}
            min={0}
            max={5_000}
            softMin={0}
            softMax={5_000}
            step={0.001}
            format={formatFixed3}
        />
    </number-control>

    <h4>Method</h4>

    <label>
        <input
            type="radio"
            name="simulation-method-type"
            bind:group={simulationState.simulationMethodType}
            value={GpuSimulationMethodType.ExplicitMpm}
        />
        Explicit MPM
    </label>

    <label>
        <input
            type="radio"
            name="simulation-method-type"
            bind:group={simulationState.simulationMethodType}
            value={GpuSimulationMethodType.MlsMpm}
        />
        MLS-MPM
    </label>

    <label>
        <input
            type="radio"
            name="simulation-method-type"
            bind:group={simulationState.simulationMethodType}
            value={GpuSimulationMethodType.FusedMlsMpm}
        />
        Fused MLS-MPM
    </label>

    <h4>Max timestep</h4>

    <number-control>
        <EntrySlider
            value={timestepDivisor}
            onValueChange={updateTimestepDivisor}
            exponential
            min={MIN_TIMESTEP_DIVISOR}
            max={MAX_TIMESTEP_DIVISOR}
            softMin={MIN_TIMESTEP_DIVISOR}
            softMax={MAX_TIMESTEP_DIVISOR}
            step={0.1}
            format={formatTimestepDivisor}
            parse={parseTimestepDivisor}
        />
    </number-control>

    {#if isTimestepCflLimited && actualTimestepDivisor !== null}
        <div>Actual timestep: 1 / {actualTimestepDivisor.toFixed(1)} s</div>
    {/if}
    
    <label>
        <input
            type="checkbox"
            bind:checked={simulationState.oneSimulationStepPerFrame}
        />
        Limit to 1 max-timestep advance per frame
    </label>
</OverlayPanel>

