<script lang="ts">
import Canvas from "$lib/Canvas.svelte";
    import Separator from "$lib/components/Separator.svelte";
    import { GpuRenderMethodType } from "$lib/gpu/pipelines/GpuRenderMethod";
    import { ElapsedTime } from "$lib/ElapsedTime.svelte";
    import ElapsedTimeDisplay from "$lib/ElapsedTimeDisplay.svelte";

let status = $state("");
let err = $state<string | null>(null);

let renderMethodType = $state(GpuRenderMethodType.Raymarch);

const elapsedTime = new ElapsedTime();
</script>

<main>
    <Canvas
        onStatusChange={text => status = text}
        onErr={text => err = text}
        {renderMethodType}
        elapsedTime={elapsedTime}
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

        <Separator />

        <h3>Elapsed time</h3>

        <dl>
            <dt>Time spent in GPU (sample)</dt>
            <dd>
                <ElapsedTimeDisplay
                    ns={elapsedTime.gpuTimeNs}
                />
            </dd>

            <dt>Total animation frame time</dt>
            <dd>
                <ElapsedTimeDisplay
                    ns={elapsedTime.animationFrameTimeNs}
                    showMsFractionalPart={false}
                />
            </dd>
        </dl>
    </overlays-panel>
</main>
ns
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