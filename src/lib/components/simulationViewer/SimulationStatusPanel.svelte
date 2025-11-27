<script lang="ts">
import type { SimulationState } from "./SimulationState.svelte";
import ElapsedTimeDisplay from "./ElapsedTimeDisplay.svelte";
import Separator from "$lib/components/generic/Separator.svelte";
import Button from "$lib/components/generic/Button.svelte";
import Hotkey from "$lib/components/headless/Hotkey.svelte";
import { GpuRenderMethodType } from "$lib/gpu/GpuRenderMethod";

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

    <!-- <div>
        <input
            type="radio"
            name="render-method-type"
            bind:group={simulationState.renderMethodType}
            value={GpuRenderMethodType.Raymarch}
            id="render-method-type_raymarch"
        />
        <label for="render-method-type_raymarch">Raymarch</label>
    </div> -->

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

    <Separator />

    <h3>Elapsed time</h3>

    <dl>
        <dt>Time spent in GPU (sample)</dt>
        <dd>
            <ElapsedTimeDisplay
                ns={simulationState.elapsedTime.gpuTimeNs}
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

    <h3>Simulation controls</h3>

    <Hotkey
        key="r"
        onKeyUp={() => simulationState.reset()}
    >
        {#snippet pressTarget({keyHeld})}
            <Button
                {keyHeld}
                onclick={() => simulationState.reset()}
            >Reset (R)</Button>
        {/snippet}
    </Hotkey>

    <Hotkey key="q">
        {#snippet pressTarget({ keyHeld })}
            {simulationState.moveForward = keyHeld}
        {/snippet}
    </Hotkey>

    <Hotkey key="e">
        {#snippet pressTarget({ keyHeld })}
            {simulationState.moveBackward = keyHeld}
        {/snippet}
    </Hotkey>

    <Hotkey key="a">
        {#snippet pressTarget({ keyHeld })}
            {simulationState.moveLeft = keyHeld}
        {/snippet}
    </Hotkey>

    <Hotkey key="d">
        {#snippet pressTarget({ keyHeld })}
            {simulationState.moveRight = keyHeld}
        {/snippet}
    </Hotkey>

    <Hotkey key="w">
        {#snippet pressTarget({ keyHeld })}
            {simulationState.moveUp = keyHeld}
        {/snippet}
    </Hotkey>

    <Hotkey key="s">
        {#snippet pressTarget({ keyHeld })}
            {simulationState.moveDown = keyHeld}
        {/snippet}
    </Hotkey>



</simulation-status-panel>

<style lang="scss">
simulation-status-panel {
    width: 20rem;
    margin: 0.5rem;
    padding: 1rem;

    line-height: 1.25;

    color: oklch(1 0 0);

    border: 2px solid oklch(1 0 0 / 0.5);
    border-radius: 2rem / 1.6rem;

    background: oklch(0 0 0 / 0.75);
}

h3 {
    font-size: 1.5rem;
    margin-bottom: 1rem;
}
</style>