<script lang="ts" module>
Chart.register(...registerables);
</script>

<script lang="ts">
import { Chart, type ChartConfiguration, registerables } from 'chart.js';
import { onMount, onDestroy, untrack } from 'svelte';


const {
    ns,
    showMsFractionalPart = true,
    inverseLabel = "Hz",
    chartMax = 1000 / 30,
}: {
    ns: bigint,
    showMsFractionalPart?: boolean,
    inverseLabel?: string,
    chartMax?: number,
} = $props();

const msWhole = $derived(ns / 1_000_000n);
const usWhole = $derived(ns / 1_000n - msWhole * 1_000n);
const nsWhole = $derived(ns - usWhole * 1_000n - msWhole * 1_000_000n);

const framesPerSecondWhole = $derived(ns === 0n ? null : 1_000_000_000n / ns)
const mframesPerSecondWhole = $derived(framesPerSecondWhole === null ? null : 1_000_000_000_000n / ns - framesPerSecondWhole * 1_000n);

let canvas: HTMLCanvasElement;
let chart: Chart | null = null;
const nHistoryEntries = 60;
let dataHistory = new Array(nHistoryEntries).fill(0);

$effect(() => {
    const ms = Number(ns) / 1_000_000;

    untrack(() => {
        dataHistory.push(ms);
        if (dataHistory.length > nHistoryEntries) {
            dataHistory.shift();
        }
        
        if (chart === null) return;

        chart.data.datasets[0].data = dataHistory;
        chart.update("none"); // no animation
    });
});

onMount(() => {
    if (!canvas) return;

    const config: ChartConfiguration = {
        type: 'line',
        data: {
            labels: new Array(nHistoryEntries).fill(''),
            datasets: [{
                data: dataHistory,
                borderColor: "oklch(1 0.01 170)",
                backgroundColor: "oklch(1 0.01 170 / 0.25)",
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.5,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    enabled: false,
                },
            },
            scales: {
                x: {
                    display: false,
                },
                y: {
                    display: false,
                    beginAtZero: true,
                    max: chartMax,
                },
            },
            elements: {
                line: {
                    borderJoinStyle: "round",
                },
            },
        },
    };

    chart = new Chart(canvas, config);
});

onDestroy(() => {
    chart?.destroy();
});
</script>

<div class="container">
    <div class="chart-container">
        <canvas bind:this={canvas}></canvas>
    </div>
    
    <elapsed-time-display>
        <elapsed-time-measurement>
            <integral-part>{msWhole}</integral-part>
            {#if showMsFractionalPart}
                <radix-point>.</radix-point>
                <fractional-part>{usWhole.toString().padStart(3, "0")}</fractional-part>
                <fractional-part>{nsWhole.toString().padStart(3, "0")}</fractional-part>
            {/if}
    
            <units-label>ms</units-label>
        </elapsed-time-measurement>
    
        <elapsed-time-measurement>
            <integral-part>{framesPerSecondWhole ?? "---"}</integral-part>
            <radix-point>.</radix-point>
            <fractional-part>{mframesPerSecondWhole?.toString().padStart(3, "0") ?? "---"}</fractional-part>
    
            <units-label>{inverseLabel}</units-label>
        </elapsed-time-measurement>
    </elapsed-time-display>
</div>

<style lang="scss">
.container {
    display: flex;
    align-items: center;
    width: 100%;
    line-height: 1.25;

    > * {
        width: 50%;
    }
}

.chart-container {
    height: 3rem;
    position: relative;
}

elapsed-time-display {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
}

elapsed-time-measurement {
    text-align: right;
    align-items: flex-end;
    font-size: 0.5rem;

    vertical-align: baseline;
    
    white-space: nowrap;
    text-align: right;
    width: 100%;
}

integral-part {
    font-size: 1.125rem;
}

radix-point {
    font-size: 0.85rem;
}

fractional-part {
    font-size: 0.85rem;
}

integral-part,
radix-point,
fractional-part {
    font-family: "Atkinson Hyperlegible Mono", monospace;
}

units-label {
    font-size: 0.85rem;

    opacity: 0.5;
}
</style>