<script lang="ts">
import type { Snippet } from "svelte";

let {
    key,
    pressTarget,
}: {
    key: string,
    pressTarget: Snippet<[{
        keyHeld: boolean,
    }]>,
} = $props();

let keyHeld = $state(false);
</script>

<svelte:window
    onkeydown={event => {
        if (event.key === key) {
            keyHeld = true;
        }
    }}
    
    onkeyup={event => {
        if (event.key === key) {
            keyHeld = false;
        }
    }}
/>

{@render pressTarget({keyHeld})}