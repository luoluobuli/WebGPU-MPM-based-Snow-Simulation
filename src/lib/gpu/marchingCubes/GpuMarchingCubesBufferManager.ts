const N_MAX_TRIANGLES = 3_500_000;
const N_MAX_VERTICES = N_MAX_TRIANGLES * 3;

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
    readonly simulationGridDims: [number, number, number];
    
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
        this.simulationGridDims = [gridResolutionX, gridResolutionY, gridResolutionZ];
        
        // Density grid uses SIMULATION resolution for consistent particle splatting
        // This is decoupled from MC grid resolution
        this.densityGridResolution = [gridResolutionX, gridResolutionY, gridResolutionZ];
        
        // MC mesh resolution (can be different from density grid)
        const mcResX = mcGridResolutionX ?? Math.floor(gridResolutionX * 1.45);
        const mcResY = mcGridResolutionY ?? Math.floor(gridResolutionY * 1.45);
        const mcResZ = mcGridResolutionZ ?? Math.floor(gridResolutionZ * 1.45);
        this.mcGridResolution = [mcResX, mcResY, mcResZ];
        
        this.vertexBuffer = device.createBuffer({
            label: "MC vertex buffer",
            size: N_MAX_VERTICES * 24,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        
        this.indirectDrawBuffer = device.createBuffer({
            label: "MC indirect draw buffer",
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        });
        
        // Atomic counter for vertices
        this.atomicCounterBuffer = device.createBuffer({
            label: "MC atomic counter buffer",
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        
        // Density grid at SIMULATION resolution (for particle splatting)
        const numDensityCells = this.densityGridResolution[0] * this.densityGridResolution[1] * this.densityGridResolution[2];
        this.densityGridBuffer = device.createBuffer({
            label: "MC density grid buffer",
            size: numDensityCells * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        
        // Vertex density/gradient at MC resolution (one extra in each dimension for corners)
        const numMcVertices = (this.mcGridResolution[0] + 1) * (this.mcGridResolution[1] + 1) * (this.mcGridResolution[2] + 1);
        this.vertexDensityBuffer = device.createBuffer({
            label: "MC vertex density buffer",
            size: numMcVertices * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        
        // Vertex gradient packed into u32 (snorm8x4)
        this.vertexGradientBuffer = device.createBuffer({
            label: "MC vertex gradient buffer",
            size: numMcVertices * 4, // u32 = 4 bytes (packed vec4f)
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Block-based optimization - uses MC resolution
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
