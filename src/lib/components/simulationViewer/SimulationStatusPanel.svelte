<script lang="ts">
import type { SimulationState } from "./SimulationState.svelte";
import ElapsedTimeDisplay from "./ElapsedTimeDisplay.svelte";
import Separator from "$lib/components/generic/Separator.svelte";
import Button from "$lib/components/generic/Button.svelte";
import Hotkey from "$lib/components/headless/Hotkey.svelte";
import { GpuRenderMethodType } from "$lib/gpu/GpuRenderMethod";
import { GpuSimulationMethodType } from "$lib/gpu/GpuSimulationMethod";

let {
    simulationState,
    status,
    err,
}: {
    simulationState: SimulationState,
    status: string,
    err: string | null,
} = $props();
</script>

<simulation-status-panel>
    <h3>Status</h3>
    <div>{err !== null ? `error: ${err}` : status}</div>

    <Separator />

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

    <Separator />

    <h3>Elapsed time</h3>

    <dl>
        <dt>GPU simulation compute pass (sample)</dt>
        <dd>
            <ElapsedTimeDisplay
                ns={simulationState.elapsedTime.gpuComputeSimulationStepTimeNs}
                inverseLabel="commands / s"
            />
        </dd>

        <dt>GPU prerender compute pass (sample)</dt>
        <dd>
            <ElapsedTimeDisplay
                ns={simulationState.elapsedTime.gpuComputePrerenderTimeNs}
                inverseLabel="commands / s"
            />
        </dd>

        <dt>GPU render pass (sample)</dt>
        <dd>
            <ElapsedTimeDisplay
                ns={simulationState.elapsedTime.gpuRenderTimeNs}
                inverseLabel="commands / s"
            />
        </dd>

        <dt>Total animation frame time</dt>
        <dd>
            <ElapsedTimeDisplay
                ns={simulationState.elapsedTime.animationFrameTimeNs}
                showMsFractionalPart={false}
            />
        </dd>
    </dl>

    <Separator />
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

    <Separator />

    <h4>Simulation method</h4>

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
    
    <Separator />

    <div>
        <input
            type="checkbox"
            bind:checked={simulationState.oneSimulationStepPerFrame}
            id="one-simulation-step-per-frame"
        />
        <label for="one-simulation-step-per-frame">Limit to 1 simulation step per frame</label>
    </div>

    <Separator />
    
    <div>
        <div>
            <h4><label for="simulation-timestep">Simulation timestep</label></h4>

            <labeled-range>
                {#if simulationState.simulationMethodType === GpuSimulationMethodType.ExplicitMpm}
                    <input
                        type="range"
                        bind:value={simulationState.explicitMpmSimulationTimestepS}
                        min={1 / 5000}
                        max={1 / 96}
                        step={1e-12}
                        id="simulation-timestep"
                    />
                {:else if simulationState.simulationMethodType === GpuSimulationMethodType.Pbmpm}
                    <input
                        type="range"
                        bind:value={simulationState.pbmpmSimulationTimestepS}
                        min={1 / 5000}
                        max={1 / 96}
                        step={1e-12}
                        id="simulation-timestep"
                    />
                {/if}

                <span>
                    {#if simulationState.simulationMethodType === GpuSimulationMethodType.ExplicitMpm}
                        <sup>1</sup>⁄<sub>{(1 / simulationState.explicitMpmSimulationTimestepS).toFixed(1)}</sub>
                    {:else if simulationState.simulationMethodType === GpuSimulationMethodType.Pbmpm}
                        <sup>1</sup>⁄<sub>{(1 / simulationState.pbmpmSimulationTimestepS).toFixed(1)}</sub>
                    {/if}
                    s
                </span>
            </labeled-range>
        </div>

    </div>

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

</simulation-status-panel>

<style lang="scss">
simulation-status-panel {
    width: 20rem;
    margin: 0.5rem;
    padding: 1rem;

    display: flex;
    align-items: stretch;
    flex-direction: column;
    gap: 0.25rem;
    overflow-y: auto;

    line-height: 1.25;

    color: oklch(1 0 0);

    border: 2px solid oklch(1 0 0 / 0.5);
    border-radius: 2rem / 1.6rem;

    background: oklch(0 0 0 / 0.75);
}

h3 {
    font-size: 1.25rem;
}

h4 {
    font-size: 1.125rem;
}

labeled-range {
    display: flex;
    align-items: center;
    gap: 1rem;
}
</style>