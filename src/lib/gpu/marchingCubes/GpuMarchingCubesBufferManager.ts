const ABSOLUTE_MAX_TRIANGLES = 3_500_000;
const ABSOLUTE_MAX_VERTICES = ABSOLUTE_MAX_TRIANGLES * 3;
const ESTIMATED_MAX_VERTICES_PER_CELL = 3;
const DEFAULT_MC_GRID_SCALE = 1 / 3;

// const MAX_MC_GRID_RES = 1_024;

export class GpuMarchingCubesBufferManager {
    readonly device: GPUDevice;
    
    readonly vertexBuffer: GPUBuffer;
    readonly indirectDrawBuffer: GPUBuffer;
    readonly atomicCounterBuffer: GPUBuffer;
    
    readonly densityGridBuffer: GPUBuffer; // density grid at simulation resolution
    
    readonly vertexDensityBuffer: GPUBuffer; // one value per MC grid vertex
    readonly vertexGradientBuffer: GPUBuffer; // for precomputed normals

    readonly activeBlocksBuffer: GPUBuffer; // for sparse update
    readonly blockIndirectDispatchBuffer: GPUBuffer;
    readonly mcGridResolution: [number, number, number];
    readonly densityGridResolution: [number, number, number];
    readonly maxVertices: number;
    
    constructor({
        device,
        gridResolutionX,
        gridResolutionY,
        gridResolutionZ,
        mcGridResolutionX,
        mcGridResolutionY,
        mcGridResolutionZ,
    }: {
        device: GPUDevice,
        gridResolutionX: number,
        gridResolutionY: number,
        gridResolutionZ: number,
        mcGridResolutionX?: number,
        mcGridResolutionY?: number,
        mcGridResolutionZ?: number,
    }) {
        this.device = device;
        
        this.densityGridResolution = [
            Math.floor(gridResolutionX * DEFAULT_MC_GRID_SCALE),
            Math.floor(gridResolutionY * DEFAULT_MC_GRID_SCALE),
            Math.floor(gridResolutionZ * DEFAULT_MC_GRID_SCALE),
        ];
        
        const mcResX = mcGridResolutionX ?? Math.floor(gridResolutionX * DEFAULT_MC_GRID_SCALE);
        const mcResY = mcGridResolutionY ?? Math.floor(gridResolutionY * DEFAULT_MC_GRID_SCALE);
        const mcResZ = mcGridResolutionZ ?? Math.floor(gridResolutionZ * DEFAULT_MC_GRID_SCALE);
        this.mcGridResolution = [mcResX, mcResY, mcResZ];

        const numMcCells = mcResX * mcResY * mcResZ;
        this.maxVertices = Math.min(
            ABSOLUTE_MAX_VERTICES,
            Math.max(1_500_000, Math.ceil(numMcCells * ESTIMATED_MAX_VERTICES_PER_CELL)),
        );
        
        this.vertexBuffer = device.createBuffer({
            label: "MC vertex buffer",
            size: this.maxVertices * 24,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        
        this.indirectDrawBuffer = device.createBuffer({
            label: "MC indirect draw buffer",
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        });
        
        this.atomicCounterBuffer = device.createBuffer({
            label: "MC atomic counter buffer",
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        
        const numDensityCells = this.densityGridResolution[0] * this.densityGridResolution[1] * this.densityGridResolution[2];
        this.densityGridBuffer = device.createBuffer({
            label: "MC density grid buffer",
            size: numDensityCells * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        
        const numMcVertices = (this.mcGridResolution[0] + 1) * (this.mcGridResolution[1] + 1) * (this.mcGridResolution[2] + 1);
        this.vertexDensityBuffer = device.createBuffer({
            label: "MC vertex density buffer",
            size: numMcVertices * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        
        this.vertexGradientBuffer = device.createBuffer({
            label: "MC vertex gradient buffer",
            size: numMcVertices * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const blocksX = Math.ceil(this.mcGridResolution[0] / 8);
        const blocksY = Math.ceil(this.mcGridResolution[1] / 8);
        const blocksZ = Math.ceil(this.mcGridResolution[2] / 8);
        const totalBlocks = blocksX * blocksY * blocksZ;

        this.activeBlocksBuffer = device.createBuffer({
            label: "MC active blocks buffer",
            size: totalBlocks * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.blockIndirectDispatchBuffer = device.createBuffer({
            label: "MC block indirect dispatch buffer",
            size: 12,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        });
    }
    
    get numCells(): number {
        return this.mcGridResolution[0] * this.mcGridResolution[1] * this.mcGridResolution[2];
    }
    
    get numVertices(): number {
        return (this.mcGridResolution[0] + 1) * (this.mcGridResolution[1] + 1) * (this.mcGridResolution[2] + 1);
    }
    
    get numDensityCells(): number {
        return this.densityGridResolution[0] * this.densityGridResolution[1] * this.densityGridResolution[2];
    }
    
    get gridDims(): [number, number, number] {
        return this.mcGridResolution;
    }


    destroy() {
        this.vertexBuffer.destroy();
        this.indirectDrawBuffer.destroy();
        this.atomicCounterBuffer.destroy();
        this.densityGridBuffer.destroy();
        this.vertexDensityBuffer.destroy();
        this.vertexGradientBuffer.destroy();
        this.activeBlocksBuffer.destroy();
        this.blockIndirectDispatchBuffer.destroy();
    }
}
