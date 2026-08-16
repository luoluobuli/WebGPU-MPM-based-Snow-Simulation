<script lang="ts">
import SimulationViewer from "$lib/components/simulationViewer/SimulationViewer.svelte";
import {
    groundScene,
    kinematicGroundFinalScene,
    kinematicGroundScene,
} from "$lib/components/simulationViewer/SimulationScene";
</script>

<SimulationViewer
    scene={groundScene}
    kinematicScene={kinematicGroundScene}
    kinematicFinalScene={kinematicGroundFinalScene}
/>
