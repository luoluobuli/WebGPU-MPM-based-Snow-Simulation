<script lang="ts">
import type { Snippet } from "svelte";

let {
    key,
    pressTarget,
    onKeyDown,
    onKeyUp,
}: {
    key: string,
    pressTarget?: Snippet<[{
        keyHeld: boolean,
    }]>,
    onKeyDown?: () => void,
    onKeyUp?: () => void,
} = $props();

let keyHeld = $state(false);
</script>

<svelte:window
    onkeydown={event => {
        if (event.key === key) {
            keyHeld = true;
            onKeyDown?.();
        }
    }}
    
    onkeyup={event => {
        if (event.key === key) {
            keyHeld = false;
            onKeyUp?.();
        }
    }}
/>

{@render pressTarget?.({keyHeld})}