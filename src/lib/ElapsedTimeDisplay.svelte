<script lang="ts">
const {
    value,
}: {
    value: bigint,
} = $props();

const msWhole = $derived(value / 1_000_000n);
const usWhole = $derived(value / 1_000n - msWhole * 1_000n);
const nsWhole = $derived(value - usWhole * 1_000n - msWhole * 1_000_000n);

const framesPerSecondWhole = $derived(value === 0n ? null : 1_000_000_000n / value)
const mframesPerSecondWhole = $derived(framesPerSecondWhole === null ? null : 1_000_000_000_000n / value - framesPerSecondWhole * 1_000n);
</script>

<elapsed-time-display>
    <elapsed-time-measurement>
        <integral-part>{msWhole}</integral-part>
        <radix-point>.</radix-point>
        <fractional-part>{usWhole.toString().padStart(3, "0")}</fractional-part>
        <fractional-part>{nsWhole.toString().padStart(3, "0")}</fractional-part>

        <units-label>ms</units-label>
    </elapsed-time-measurement>

    <elapsed-time-measurement>
        <integral-part>{framesPerSecondWhole ?? "---"}</integral-part>
        <radix-point>.</radix-point>
        <fractional-part>{mframesPerSecondWhole?.toString().padStart(3, "0") ?? "---"}</fractional-part>

        <units-label>fps</units-label>
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

integral-part,
radix-point,
fractional-part,
fractional-part {
    font-family: monospace;
}
</style>