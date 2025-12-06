<script lang="ts">
    import type { Snippet } from "svelte";

let {
    children,
}: {
    children?: Snippet,
} = $props();
</script>

<overlay-panel>
    {@render children?.()}
</overlay-panel>


<style lang="scss">
overlay-panel {
    width: 20rem;
    margin: 0.5rem;
    padding: 1rem;

    display: flex;
    align-items: stretch;
    flex-direction: column;
    gap: 0.5rem;
    overflow-y: auto;

    line-height: 1.25;

    color: oklch(1 0 0);

    border: 2px solid oklch(1 0 0 / 0.5);
    border-radius: 2rem / 1.6rem;

    background: oklch(0 0 0 / 0.75);

    :global(h3) {
        margin-top: 0.5rem;
        font-size: 1.5rem;
    }

    :global(h4) {
        margin-top: 0.5rem;
        font-size: 1.125rem;
    }

    :global(input[type="range"]) {
        width: 10.5rem;
    }

    :global(labeled-range) {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
}
</style>