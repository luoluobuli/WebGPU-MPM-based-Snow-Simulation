<script lang="ts">
import Canvas from "$lib/Canvas.svelte";

let status = $state("");
let err = $state<string | null>(null);

$inspect(status);
</script>

<main>
    <Canvas
        onStatusChange={text => status = text}
        onErr={text => err = text}
    />

    <overlays-panel>
        {err !== null ? `error: ${err}` : status}
    </overlays-panel>
</main>

<style lang="scss">
main {
    width: 100vw;
    height: 100vh;

    display: grid;

    overlays-panel {
        width: 20%;
        padding: 0.5rem;
        
        color: oklch(1 0 0);
    }

    > :global(*) {
        grid-area: 1/1;
    }
}
</style>