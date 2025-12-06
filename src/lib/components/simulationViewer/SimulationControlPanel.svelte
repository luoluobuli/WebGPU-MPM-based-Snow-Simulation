<script lang="ts">
import type { SimulationState } from "./SimulationState.svelte";
import Separator from "$lib/components/generic/Separator.svelte";
import Button from "$lib/components/generic/Button.svelte";
import Hotkey from "$lib/components/headless/Hotkey.svelte";
import { GpuRenderMethodType } from "$lib/gpu/GpuRenderMethod";
import { GpuSimulationMethodType } from "$lib/gpu/GpuSimulationMethod";
import OverlayPanel from "./OverlayPanel.svelte";

let {
    simulationState,
}: {
    simulationState: SimulationState,
} = $props();


const MIN_TIMESTEP_DIVISOR = 30;
const MAX_TIMESTEP_DIVISOR = 10_000;


let timestepProgress = $derived.by(() => {
    switch (simulationState.simulationMethodType) {
        case GpuSimulationMethodType.ExplicitMpm:
            return 1 - Math.log2((1 / simulationState.explicitMpmSimulationTimestepS) / MIN_TIMESTEP_DIVISOR) / (MAX_TIMESTEP_DIVISOR / MIN_TIMESTEP_DIVISOR);

        case GpuSimulationMethodType.Pbmpm:
            return 1 - Math.log2((1 / simulationState.pbmpmSimulationTimestepS) / MIN_TIMESTEP_DIVISOR) / (MAX_TIMESTEP_DIVISOR / MIN_TIMESTEP_DIVISOR);
    }
});

const timestepDivisor = $derived(Math.pow(MAX_TIMESTEP_DIVISOR / MIN_TIMESTEP_DIVISOR, 1 - timestepProgress) * MIN_TIMESTEP_DIVISOR);
const timestep = $derived(1 / timestepDivisor);

const updateTimestep = () => {
    switch (simulationState.simulationMethodType) {
        case GpuSimulationMethodType.ExplicitMpm:
            simulationState.explicitMpmSimulationTimestepS = timestep;
            break;

        case GpuSimulationMethodType.Pbmpm:
            simulationState.pbmpmSimulationTimestepS = timestep;
            break;
    }
};
</script>

<OverlayPanel>
    <h3>Render method</h3>

    <div>
        <input
            type="radio"
            name="render-method-type"
            bind:group={simulationState.renderMethodType}
            value={GpuRenderMethodType.Points}
            id="render-method-type_points"
        />
        <label for="render-method-type_points">Points</label>
    </div>

    <div>
        <input
            type="radio"
            name="render-method-type"
            bind:group={simulationState.renderMethodType}
            value={GpuRenderMethodType.Volumetric}
            id="render-method-type_volumetric"
        />
        <label for="render-method-type_volumetric">Volumetric</label>
    </div>

    <div>
        <input
            type="radio"
            name="render-method-type"
            bind:group={simulationState.renderMethodType}
            value={GpuRenderMethodType.Ssfr}
            id="render-method-type_ssfr"
        />
        <label for="render-method-type_ssfr">SSFR</label>
    </div>

    <div>
        <input
            type="radio"
            name="render-method-type"
            bind:group={simulationState.renderMethodType}
            value={GpuRenderMethodType.MarchingCubes}
            id="render-method-type_marching-cubes"
        />
        <label for="render-method-type_marching-cubes">Marching cubes</label>
    </div>

    <Separator />

    <h3>Simulation controls</h3>

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

    <h4>Method</h4>

    <div>
        <input
            type="radio"
            name="simulation-method-type"
            bind:group={simulationState.simulationMethodType}
            value={GpuSimulationMethodType.ExplicitMpm}
            id="simulation-method-type_explicit-mpm"
        />
        <label for="simulation-method-type_explicit-mpm">Explicit MPM</label>
    </div>

    <div>
        <input
            type="radio"
            name="simulation-method-type"
            bind:group={simulationState.simulationMethodType}
            value={GpuSimulationMethodType.Pbmpm}
            id="simulation-method-type_pbmpm"
        />
        <label for="simulation-method-type_pbmpm">PBMPM</label>
    </div>

    <div>
        <h4>Timestep</h4>

        <labeled-range>
            <input
                type="range"
                bind:value={timestepProgress}
                oninput={() => updateTimestep()}
                min={0}
                max={1}
                step={Number.EPSILON}
                id="simulation-timestep"
            />

            <span>
                <sup>1</sup>⁄<sub>{timestepDivisor.toFixed(1)}</sub>
                s
            </span>
        </labeled-range>
    </div>
    
    <div>
        <input
            type="checkbox"
            bind:checked={simulationState.oneSimulationStepPerFrame}
            id="one-simulation-step-per-frame"
        />
        <label for="one-simulation-step-per-frame">Limit to 1 simulation step per frame</label>
    </div>
</OverlayPanel>

<Hotkey key="q"
    onKeyUp={() => simulationState.moveForward = false}
    onKeyDown={() => simulationState.moveForward = true}
/>

<Hotkey key="e"
    onKeyUp={() => simulationState.moveBackward = false}
    onKeyDown={() => simulationState.moveBackward = true}
/>

<Hotkey key="a"
    onKeyUp={() => simulationState.moveLeft = false}
    onKeyDown={() => simulationState.moveLeft = true}
/>

<Hotkey key="d"
    onKeyUp={() => simulationState.moveRight = false}
    onKeyDown={() => simulationState.moveRight = true}
/>

<Hotkey key="w"
    onKeyUp={() => simulationState.moveUp = false}
    onKeyDown={() => simulationState.moveUp = true}
/>

<Hotkey key="s"
    onKeyUp={() => simulationState.moveDown = false}
    onKeyDown={() => simulationState.moveDown = true}
/>