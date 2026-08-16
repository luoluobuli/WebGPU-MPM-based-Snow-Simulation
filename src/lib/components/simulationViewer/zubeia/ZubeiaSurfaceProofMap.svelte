<script module lang="ts">
import type { ZubeiaMatchedTopologyProof } from "./ZubeiaMatchedTopologyProof";
import type { ZubeiaPawContactMask } from "./ZubeiaPawContactMask";

type Rgb = readonly [number, number, number];

export const ZUBEIA_MATCHED_MAP_DEPTH_MAX_M = 0.075;
export const ZUBEIA_MATCHED_MAP_PIXELS_PER_CELL = 4;
export const ZUBEIA_MATCHED_MAP_COLORS = {
    noMatchedSurface: [72, 80, 88],
    missingColliderSurface: [0, 0, 0],
    pawContact: [255, 215, 88],
    pawPerimeter: [66, 221, 244],
    qualifyingCrack: [242, 48, 70],
    controlOpening: [198, 88, 246],
} as const satisfies Record<string, Rgb>;

const DEPTH_COLOR_STOPS: readonly Rgb[] = [
    [205, 222, 235],
    [53, 190, 240],
    [62, 201, 123],
    [250, 184, 62],
    [235, 63, 79],
];

const mixChannel = (
    first: Rgb,
    second: Rgb,
    channel: number,
    alpha: number,
) => Math.round(
    first[channel] * (1 - alpha) + second[channel] * alpha,
);

const mixColor = (
    first: Rgb,
    second: Rgb,
    alpha: number,
): Rgb => [
    mixChannel(first, second, 0, alpha),
    mixChannel(first, second, 1, alpha),
    mixChannel(first, second, 2, alpha),
];

export const zubeiaMatchedDepthColor = (depthM: number): Rgb => {
    const scaled = Math.min(
        1,
        Math.max(0, depthM / ZUBEIA_MATCHED_MAP_DEPTH_MAX_M),
    ) * (DEPTH_COLOR_STOPS.length - 1);
    const firstIndex = Math.floor(scaled);
    const secondIndex = Math.min(
        DEPTH_COLOR_STOPS.length - 1,
        firstIndex + 1,
    );

    return mixColor(
        DEPTH_COLOR_STOPS[firstIndex],
        DEPTH_COLOR_STOPS[secondIndex],
        scaled - firstIndex,
    );
};

const assertCellArray = (
    values: ArrayLike<number>,
    label: string,
    cellCount: number,
) => {
    if (values.length === cellCount) return;

    throw new Error(
        `Zubeia matched ${label} must contain ${cellCount} cells`,
    );
};

const setPixel = (
    pixels: Uint8ClampedArray,
    width: number,
    x: number,
    y: number,
    color: Rgb,
) => {
    const offset = (y * width + x) * 4;
    pixels[offset] = color[0];
    pixels[offset + 1] = color[1];
    pixels[offset + 2] = color[2];
    pixels[offset + 3] = 255;
};

export const renderZubeiaMatchedSurfacePixels = ({
    proof,
    pawContactMask,
}: {
    proof: ZubeiaMatchedTopologyProof,
    pawContactMask: ZubeiaPawContactMask,
}) => {
    const map = proof.surfaceProofMap;
    if (
        map.walkingCellCount !== pawContactMask.walkingCellCount
        || map.crossStripCellCount !== pawContactMask.crossStripCellCount
    ) {
        throw new Error("Zubeia matched proof and paw mask dimensions must match");
    }

    const cellCount = map.walkingCellCount * map.crossStripCellCount;
    for (const [label, values] of [
        ["collider presence", map.colliderSurfacePresent],
        ["control presence", map.controlSurfacePresent],
        ["matched presence", map.matchedSurfacePresent],
        ["collider top Z", map.colliderTopZM],
        ["control top Z", map.controlTopZM],
        ["depression", map.matchedDepressionM],
        ["paw contact", pawContactMask.contact],
        ["paw perimeter", pawContactMask.perimeter],
        ["qualifying crack", proof.crackCellMask],
        ["control opening", proof.controlOpeningCellMask],
    ] as const) {
        assertCellArray(values, label, cellCount);
    }

    const pixelsPerCell = ZUBEIA_MATCHED_MAP_PIXELS_PER_CELL;
    const width = map.walkingCellCount * pixelsPerCell;
    const height = map.crossStripCellCount * pixelsPerCell;
    const pixels = new Uint8ClampedArray(width * height * 4);

    for (let crossIndex = 0; crossIndex < map.crossStripCellCount; crossIndex++) {
        for (
            let walkingIndex = 0;
            walkingIndex < map.walkingCellCount;
            walkingIndex++
        ) {
            const mapIndex = crossIndex * map.walkingCellCount + walkingIndex;
            const colliderPresent = map.colliderSurfacePresent[mapIndex] !== 0;
            const controlPresent = map.controlSurfacePresent[mapIndex] !== 0;
            let baseColor: Rgb = ZUBEIA_MATCHED_MAP_COLORS.noMatchedSurface;
            if (controlPresent && !colliderPresent) {
                baseColor = ZUBEIA_MATCHED_MAP_COLORS.missingColliderSurface;
            } else if (controlPresent && colliderPresent) {
                const depressionM = Math.max(
                    0,
                    map.controlTopZM[mapIndex] - map.colliderTopZM[mapIndex],
                );
                baseColor = zubeiaMatchedDepthColor(depressionM);
            }

            const contact = pawContactMask.contact[mapIndex] !== 0;
            const perimeter = pawContactMask.perimeter[mapIndex] !== 0;
            const crack = proof.crackCellMask[mapIndex] !== 0;
            const controlOpening = proof.controlOpeningCellMask[mapIndex] !== 0;
            const contactTint = mixColor(
                baseColor,
                ZUBEIA_MATCHED_MAP_COLORS.pawContact,
                0.32,
            );
            const imageCellY = map.crossStripCellCount - crossIndex - 1;

            for (let localY = 0; localY < pixelsPerCell; localY++) {
                for (let localX = 0; localX < pixelsPerCell; localX++) {
                    let color = contact ? contactTint : baseColor;
                    if (contact && (localX + localY) % pixelsPerCell === 0) {
                        color = ZUBEIA_MATCHED_MAP_COLORS.pawContact;
                    }
                    if (
                        perimeter
                        && (
                            localX === 0
                            || localX === pixelsPerCell - 1
                            || localY === 0
                            || localY === pixelsPerCell - 1
                        )
                    ) {
                        color = ZUBEIA_MATCHED_MAP_COLORS.pawPerimeter;
                    }
                    if (
                        crack
                        && (
                            localX === localY
                            || localX + localY === pixelsPerCell - 1
                        )
                    ) {
                        color = ZUBEIA_MATCHED_MAP_COLORS.qualifyingCrack;
                    }
                    if (controlOpening && (localX + localY) % 2 === 0) {
                        color = ZUBEIA_MATCHED_MAP_COLORS.controlOpening;
                    }

                    setPixel(
                        pixels,
                        width,
                        walkingIndex * pixelsPerCell + localX,
                        imageCellY * pixelsPerCell + localY,
                        color,
                    );
                }
            }
        }
    }

    return {
        width,
        height,
        pixels,
    };
};
</script>

<script lang="ts">
let {
    proof,
    pawContactMask,
}: {
    proof: ZubeiaMatchedTopologyProof,
    pawContactMask: ZubeiaPawContactMask,
} = $props();

let canvas = $state<HTMLCanvasElement | null>(null);

$effect(() => {
    const target = canvas;
    const rendered = renderZubeiaMatchedSurfacePixels({
        proof,
        pawContactMask,
    });
    if (target === null) return;

    target.width = rendered.width;
    target.height = rendered.height;
    const context = target.getContext("2d");
    if (context === null) return;

    const image = context.createImageData(rendered.width, rendered.height);
    image.data.set(rendered.pixels);
    context.putImageData(image, 0, 0);
});
</script>

<figure>
    <canvas
        bind:this={canvas}
        width={proof.surfaceProofMap.walkingCellCount * ZUBEIA_MATCHED_MAP_PIXELS_PER_CELL}
        height={proof.surfaceProofMap.crossStripCellCount * ZUBEIA_MATCHED_MAP_PIXELS_PER_CELL}
        aria-label="Collider-minus-control matched Zubeia ground map; walk direction is left to right"
    ></canvas>
    <figcaption>
        <div class="direction">walk direction -&gt;</div>
        <div class="depth-legend">
            <span>matched depression</span>
            <span class="depth-gradient"></span>
            <span>0</span>
            <span>75+ mm</span>
        </div>
        <ul class="overlay-legend" aria-label="Matched proof overlays">
            <li><span class="swatch missing"></span>missing collider surface</li>
            <li><span class="swatch contact"></span>exact paw contact</li>
            <li><span class="swatch perimeter"></span>paw perimeter</li>
            <li><span class="swatch crack"></span>qualifying crack</li>
            <li><span class="swatch contaminated"></span>control opening</li>
        </ul>
        <p>
            All evidence is collider minus its matched no-collider control.
            Depression is positive downward: control top Z minus collider top
            Z, clamped at 75 mm. Black means the control has crust where the
            collider run has none. Gray has no matched control surface. Red
            marks qualifying crack cells; purple marks control-run opening and
            is not crack evidence.
        </p>
    </figcaption>
</figure>

<style lang="scss">
figure {
    margin: 0;

    display: grid;
    gap: 0.3rem;
}

canvas {
    display: block;
    width: 100%;
    height: auto;

    border: 0.0625rem solid oklch(0.84 0.02 230 / 0.65);
    background: oklch(0.3 0.01 235);

    image-rendering: pixelated;
}

figcaption {
    display: grid;
    gap: 0.25rem;

    color: oklch(0.84 0.02 230);
    font-size: 0.72rem;
}

.direction {
    text-align: center;
}

.depth-legend {
    display: grid;
    grid-template-columns: auto minmax(2rem, 1fr) auto auto;
    align-items: center;
    gap: 0.3rem;
}

.depth-gradient {
    height: 0.45rem;

    background: linear-gradient(
        90deg,
        oklch(0.88 0.03 230),
        oklch(0.72 0.15 225),
        oklch(0.74 0.16 155),
        oklch(0.8 0.15 75),
        oklch(0.64 0.21 20)
    );
}

.overlay-legend {
    margin: 0;
    padding: 0;

    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem 0.7rem;

    list-style: none;
}

.overlay-legend li {
    display: inline-flex;
    align-items: center;
    gap: 0.24rem;
}

.swatch {
    width: 0.72rem;
    height: 0.72rem;

    border: 0.0625rem solid oklch(0.9 0.01 230 / 0.7);
}

.missing {
    background: oklch(0 0 0);
}

.contact {
    background: repeating-linear-gradient(
        135deg,
        oklch(0.87 0.17 90) 0 0.12rem,
        oklch(0.48 0.03 230) 0.12rem 0.24rem
    );
}

.perimeter {
    background: oklch(0.84 0.14 210);
}

.crack {
    background: oklch(0.62 0.24 20);
}

.contaminated {
    background: oklch(0.65 0.23 315);
}

p {
    margin: 0;
}
</style>
