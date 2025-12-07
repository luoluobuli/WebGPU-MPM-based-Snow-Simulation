<script lang="ts">
const {
    ns,
    showMsFractionalPart = true,
    inverseLabel = "Hz",
}: {
    ns: bigint,
    showMsFractionalPart?: boolean,
    inverseLabel?: string,
} = $props();

const msWhole = $derived(ns / 1_000_000n);
const usWhole = $derived(ns / 1_000n - msWhole * 1_000n);
const nsWhole = $derived(ns - usWhole * 1_000n - msWhole * 1_000_000n);

const framesPerSecondWhole = $derived(ns === 0n ? null : 1_000_000_000n / ns)
const mframesPerSecondWhole = $derived(framesPerSecondWhole === null ? null : 1_000_000_000_000n / ns - framesPerSecondWhole * 1_000n);
</script>

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

<fps-display>

</fps-display>

<style lang="scss">
elapsed-time-display {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
}

elapsed-time-measurement {
    display: flex;
    gap: 0.25rem;
    align-items: flex-end;
    

    text-align: right;
}

integral-part {
    font-size: 1.25rem;
}

fractional-part {
    font-size: 1rem;
}

integral-part,
radix-point,
fractional-part {
    font-family: "Atkinson Hyperlegible Mono", monospace;
}

units-label {
    opacity: 0.5;
}
</style>