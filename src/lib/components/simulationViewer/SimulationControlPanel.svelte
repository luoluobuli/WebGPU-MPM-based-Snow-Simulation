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
            return progressFromTimestep(simulationState.explicitMpmSimulationTimestepS);

        case GpuSimulationMethodType.Pbmpm:
            return progressFromTimestep(simulationState.pbmpmSimulationTimestepS);
    }
});

const timestepDivisor = $derived(Math.pow(MAX_TIMESTEP_DIVISOR / MIN_TIMESTEP_DIVISOR, 1 - timestepProgress) * MIN_TIMESTEP_DIVISOR);

const updateTimestep = (progress: number) => {
    switch (simulationState.simulationMethodType) {
        case GpuSimulationMethodType.ExplicitMpm:
            simulationState.explicitMpmSimulationTimestepS = 1 / (Math.pow(MAX_TIMESTEP_DIVISOR / MIN_TIMESTEP_DIVISOR, 1 - progress) * MIN_TIMESTEP_DIVISOR);
            break;

        case GpuSimulationMethodType.Pbmpm:
            simulationState.pbmpmSimulationTimestepS = 1 / (Math.pow(MAX_TIMESTEP_DIVISOR / MIN_TIMESTEP_DIVISOR, 1 - progress) * MIN_TIMESTEP_DIVISOR);
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
            value={GpuSimulationMethodType.Pbmpm}
        />
        PBMPM
    </label>

    <h4>Timestep</h4>

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
            <sup>1</sup>⁄<sub>{timestepDivisor.toFixed(1)}</sub>
            s
        </span>
    </labeled-range>
    
    <label>
        <input
            type="checkbox"
            bind:checked={simulationState.oneSimulationStepPerFrame}
        />
        Limit to 1 simulation step per frame
    </label>
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