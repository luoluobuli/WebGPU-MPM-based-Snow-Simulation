<script lang="ts">
import Canvas from "$lib/Canvas.svelte";
    import Separator from "$lib/components/Separator.svelte";
    import { GpuRenderMethodType } from "$lib/gpu/pipelines/GpuRenderMethod";

let status = $state("");
let err = $state<string | null>(null);

let renderMethodType = $state(GpuRenderMethodType.Points);
</script>

<main>
    <Canvas
        onStatusChange={text => status = text}
        onErr={text => err = text}
        {renderMethodType}
    />

    <overlays-panel>
        <h3>Status</h3>
        <div>{err !== null ? `error: ${err}` : status}</div>

        <Separator />

        <h3>Render method</h3>

        <div>
            <input
                type="radio"
                name="render-method-type"
                bind:group={renderMethodType}
                value={GpuRenderMethodType.Points}
                id="render-method-type_points"
            />
            <label for="render-method-type_points">Points</label>
        </div>

        <div>
            <input
                type="radio"
                name="render-method-type"
                bind:group={renderMethodType}
                value={GpuRenderMethodType.Raymarch}
                id="render-method-type_raymarch"
            />
            <label for="render-method-type_raymarch">Raymarch</label>
        </div>
    </overlays-panel>
</main>

<style lang="scss">
main {
    width: 100vw;
    height: 100vh;

    display: grid;

    > :global(*) {
        grid-area: 1/1;
    }
}

overlays-panel {
    width: 20%;
    padding: 0.5rem;

    color: oklch(1 0 0);
}
</style>