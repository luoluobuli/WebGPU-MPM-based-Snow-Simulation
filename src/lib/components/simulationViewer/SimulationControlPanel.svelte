<script lang="ts">
import type { SimulationState } from "./SimulationState.svelte";
import Separator from "$lib/components/generic/Separator.svelte";
import Button from "$lib/components/generic/Button.svelte";
import Hotkey from "$lib/components/headless/Hotkey.svelte";
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

let timestepProgress = $derived.by(() => {
    switch (simulationState.simulationMethodType) {
        case GpuSimulationMethodType.ExplicitMpm:
            return progressFromTimestep(simulationState.explicitMpmMaxSimulationTimestepS);

        case GpuSimulationMethodType.MlsMpm:
            return progressFromTimestep(simulationState.mlsMpmMaxSimulationTimestepS);
    }
});

const timestepDivisor = $derived(Math.pow(MAX_TIMESTEP_DIVISOR / MIN_TIMESTEP_DIVISOR, 1 - timestepProgress) * MIN_TIMESTEP_DIVISOR);
const actualTimestepDivisor = $derived(simulationState.actualSimulationTimestepS === null
    ? null
    : 1 / simulationState.actualSimulationTimestepS);
const isTimestepCflLimited = $derived(actualTimestepDivisor !== null && actualTimestepDivisor > timestepDivisor * 1.001);

const updateTimestep = (progress: number) => {
    switch (simulationState.simulationMethodType) {
        case GpuSimulationMethodType.ExplicitMpm:
            simulationState.explicitMpmMaxSimulationTimestepS = 1 / (Math.pow(MAX_TIMESTEP_DIVISOR / MIN_TIMESTEP_DIVISOR, 1 - progress) * MIN_TIMESTEP_DIVISOR);
            break;

        case GpuSimulationMethodType.MlsMpm:
            simulationState.mlsMpmMaxSimulationTimestepS = 1 / (Math.pow(MAX_TIMESTEP_DIVISOR / MIN_TIMESTEP_DIVISOR, 1 - progress) * MIN_TIMESTEP_DIVISOR);
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

    <div>Right-click the snow to:</div>

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

    <labeled-range>
        <input
            type="range"
            bind:value={simulationState.interactionRadiusFactor}
            min={0}
            max={15}
            step={Number.EPSILON}
        />

        <span>{simulationState.interactionRadiusFactor.toFixed(3)}</span>
    </labeled-range>

    <div>Interaction strength</div>

    <labeled-range>
        <input
            type="range"
            bind:value={simulationState.interactionStrength}
            min={0}
            max={5_000}
            step={Number.EPSILON}
        />

        <span>{simulationState.interactionStrength.toFixed(3)}</span>
    </labeled-range>

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

    <h4>Max timestep</h4>

    <labeled-range>
        <input
            type="range"
            bind:value={timestepProgress}
            oninput={() => updateTimestep(timestepProgress)}
            min={0}
            max={1}
            step={Number.EPSILON}
        />

        <span>
            max 1 / {timestepDivisor.toFixed(1)}
            s
        </span>
    </labeled-range>

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

