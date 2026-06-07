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
    padding: 0.75rem;

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
        font-size: 1.25rem;
    }

    :global(h4) {
        margin-top: 0.5rem;
        font-size: 1rem;
    }

    :global(entry-slider) {
        width: 10.5rem;
    }

    :global(entry-slider > input) {
        width: 100%;
        min-height: 1.75rem;
        padding: 0.2rem 0.45rem;

        appearance: none;

        color: oklch(0.98 0.01 220);
        background:
            linear-gradient(
                90deg,
                oklch(0.72 0.12 215 / 0.5) 0 calc(var(--entry-slider-progress, 0) * 100%),
                oklch(0.16 0.02 230 / 0.92) calc(var(--entry-slider-progress, 0) * 100%) 100%
            );
        border: 1px solid oklch(0.92 0.01 220 / 0.42);
        border-radius: 0.25rem;
        outline: none;
        box-shadow: inset 0 0 0 1px oklch(0 0 0 / 0.25);

        font-variant-numeric: tabular-nums;
        text-align: right;

        cursor: ew-resize;

        transition:
            background-color 120ms ease,
            border-color 120ms ease,
            box-shadow 120ms ease,
            color 120ms ease;
    }

    :global(entry-slider > input:hover),
    :global(entry-slider:focus-within > input) {
        border-color: oklch(0.82 0.13 215);
        box-shadow:
            inset 0 0 0 1px oklch(0 0 0 / 0.25),
            0 0 0 0.12rem oklch(0.72 0.12 215 / 0.35);
    }

    :global(entry-slider.dragging > input) {
        border-color: oklch(0.78 0.16 155);
        box-shadow:
            inset 0 0 0 1px oklch(0 0 0 / 0.25),
            0 0 0 0.12rem oklch(0.7 0.14 155 / 0.3);
    }

    :global(entry-slider.editing > input) {
        border-color: oklch(0.82 0.13 215);
        box-shadow:
            inset 0 0 0 1px oklch(0 0 0 / 0.25),
            0 0 0 0.12rem oklch(0.72 0.12 215 / 0.35);

        cursor: text;
    }

    :global(entry-slider.outside-soft-bounds > input) {
        border-color: oklch(0.78 0.14 85);
        color: oklch(0.9 0.12 85);
    }

    :global(entry-slider.outside-hard-bounds > input) {
        border-color: oklch(0.68 0.2 25);
        color: oklch(0.78 0.18 25);
    }

    :global(entry-slider.disabled > input) {
        opacity: 0.58;
        cursor: not-allowed;
    }

    :global(number-control) {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
}
</style>
