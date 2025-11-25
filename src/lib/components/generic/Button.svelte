<script lang="ts">
    import type { Snippet } from "svelte";
    import type { HTMLButtonAttributes } from "svelte/elements";

let {
    children,
    keyHeld = false,
    ...buttonProps
}: {
    children: Snippet,
    keyHeld?: boolean,
} & HTMLButtonAttributes = $props();
</script>

<button {...buttonProps}>
    <button-display class:key-held={keyHeld}>
        {@render children?.()}
    </button-display>
</button>

<style lang="scss">
button {
    margin: 0;
    padding: 0;

    border: none;
    background: none;

    cursor: pointer;

    &:is(:hover, :focus-visible) > button-display,
    button-display.key-held {
        border-color: currentcolor;

        transform: translateY(-0.125rem) scale(1.1);
        box-shadow: 0 0.25rem 0.5rem oklch(from currentcolor l c h / 0.5);

        animation: sliding-background 1s infinite linear;
        background: repeating-linear-gradient(
            135deg,
            oklch(0 0 0 / 0) 0,
            oklch(0 0 0 / 0) 1rem,
            oklch(from currentcolor l c h / 0.25) 1rem,
            oklch(from currentcolor l c h / 0.25) 2rem,
        );
        background-size: calc(100% + 3rem) calc(100% + 3rem);

        @keyframes sliding-background {
            from {
                background-position: -2.828427rem -2.828427rem; // 2 * sqrt(2)
            }
            to {
                background-position: 0 0;
            }
        }
    }

    &:active > button-display,
    button-display.key-held {
        transform: translateY(0.0625rem) scale(0.95);

        box-shadow: 0 0 0 oklch(0 0 0 / 0);
        filter: brightness(0.5);
    }
}

button-display {
    display: block;

    padding: 0.25rem 1rem;

    border: 2px solid oklch(from currentcolor l c h / 0.5);
    border-radius: 1rem;

    pointer-events: none;

    transition:
        transform 0.25s cubic-bezier(0,2.75,.47,1),
        box-shadow 0.1s cubic-bezier(0,1,.47,1),
        filter 0.25s ease;
}
</style>