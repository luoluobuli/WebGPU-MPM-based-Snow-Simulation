<script lang="ts">
import SimulationViewer from "$lib/components/simulationViewer/SimulationViewer.svelte";
import {
    groundScene,
    kinematicGroundScene,
} from "$lib/components/simulationViewer/SimulationScene";
</script>

<SimulationViewer
    scene={groundScene}
    kinematicScene={kinematicGroundScene}
/>