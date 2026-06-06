export enum ParticleAppearanceMaterial {
    Default = 0,
    Soil = 1,
    Bark = 2,
    Leaf = 3,
}

const clampByte = (value: number) =>
    Math.max(0, Math.min(255, Math.round(value * 255)));

export const packParticleAppearance = ({
    color,
    material,
}: {
    color: [number, number, number],
    material: ParticleAppearanceMaterial,
}) => {
    const r = clampByte(color[0]);
    const g = clampByte(color[1]);
    const b = clampByte(color[2]);

    return (material << 24) | (b << 16) | (g << 8) | r;
};

export class GpuParticleAppearanceBufferManager {
    readonly appearanceBuffer: GPUBuffer;

    constructor({
        device,
        nParticles,
        appearances,
    }: {
        device: GPUDevice,
        nParticles: number,
        appearances?: Uint32Array | null,
    }) {
        const particleAppearances = appearances ?? new Uint32Array(nParticles);

        if (particleAppearances.length !== nParticles) {
            throw new Error("particle appearance count must match particle count");
        }

        const buffer = device.createBuffer({
            label: "particle appearance buffer",
            size: Math.max(particleAppearances.byteLength, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        if (particleAppearances.byteLength > 0) {
            device.queue.writeBuffer(
                buffer,
                0,
                particleAppearances.buffer as ArrayBuffer,
                particleAppearances.byteOffset,
                particleAppearances.byteLength,
            );
        }

        this.appearanceBuffer = buffer;
    }
}
