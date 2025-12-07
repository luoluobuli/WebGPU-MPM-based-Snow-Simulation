export class PrerenderPassElapsedTime {
    readonly label: string = $state()!;
    elapsedTimeNs: bigint | null = $state(null);

    constructor(label: string) {
        this.label = label;
    }
}