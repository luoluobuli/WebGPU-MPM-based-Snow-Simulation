<script lang="ts">
import Canvas from "$lib/Canvas.svelte";

let status = $state("");
let err = $state<string | null>();
</script>

<Canvas
    onStatusChange={text => status = text}
    onErr={text => err = text}
/>
