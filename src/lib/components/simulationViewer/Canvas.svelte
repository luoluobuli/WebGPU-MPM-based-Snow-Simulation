<script lang="ts">
import Draggable from "$lib/components/headless/Draggable.svelte";
import type { SimulationState } from "./SimulationState.svelte";

let {
    simulationState,
    canvas = $bindable(),
}: {
    simulationState: SimulationState,
    canvas?: HTMLCanvasElement | null;
} = $props();
</script>

<section
    bind:clientWidth={null, clientWidth => simulationState.width = clientWidth!}
    bind:clientHeight={null, clientHeight => simulationState.height = clientHeight!}
>
    <Draggable
        onDown={({ pointerEvent }) => {
            if (pointerEvent.button === 2) {
                // Right click: Interact
                simulationState.onInteractionStart(pointerEvent.clientX, pointerEvent.clientY, canvas!);
            } else {
                // Left/Middle: Camera
                canvas?.requestPointerLock();
            }
        }}

        onDrag={async ({ movement, button, pointerEvent }) => {
            switch (button) {
                case 0:
                    simulationState.orbit.turn(movement);
                    break;

                case 1:
                    simulationState.orbit.pan(movement);
                    break;
                
                case 2:
                    simulationState.onInteractionDrag(pointerEvent.clientX, pointerEvent.clientY, canvas!);
                    break;
            }

            pointerEvent.preventDefault();
        }}

        onUp={({ pointerEvent }) => {
            if (pointerEvent.button === 2) {
                simulationState.onInteractionEnd();
            } else {
                document.exitPointerLock();
            }
        }}
    >
        {#snippet dragTarget({ onpointerdown })}
            <canvas
                bind:this={canvas}
                width={simulationState.width}
                height={simulationState.height}
                {onpointerdown}
                oncontextmenu={(e) => { e.preventDefault(); }}
                onwheel={(event) => {
                    simulationState.orbit.radius *= 2 ** (event.deltaY * 0.001);
                    event.preventDefault();
                }}
            ></canvas>
        {/snippet}
    </Draggable>
</section>

<style lang="scss">
    section {
        width: 100vw;
        height: 100vh;
    }
</style>
