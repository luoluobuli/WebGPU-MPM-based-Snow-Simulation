const N_MAX_TRIANGLES = 3_500_000;
const N_MAX_VERTICES = N_MAX_TRIANGLES * 3;

// const MAX_MC_GRID_RES = 1_024;

export class GpuMarchingCubesBufferManager {
    readonly device: GPUDevice;
    
    // Vertex buffer: each vertex has position (vec3f) + normal (vec3f) = 24 bytes
    readonly vertexBuffer: GPUBuffer;
    
    // Indirect draw buffer: vertexCount, instanceCount, firstVertex, firstInstance
    readonly indirectDrawBuffer: GPUBuffer;
    
    // Atomic counter for vertex allocation
    readonly atomicCounterBuffer: GPUBuffer;
    
    readonly densityGridBuffer: GPUBuffer; // need our own smoothed version of the mpm mass grid
    
    readonly vertexDensityBuffer: GPUBuffer; // one value per grid vertex
    readonly vertexGradientBuffer: GPUBuffer; // for precomputed normals

    readonly activeBlocksBuffer: GPUBuffer; // for sparse update
    readonly blockIndirectDispatchBuffer: GPUBuffer;
    readonly gridResolution: [number, number, number];
    readonly simulationGridDims: [number, number, number];
    // readonly downsampleFactor: number;
    
    constructor({
        device,
        gridResolutionX,
        gridResolutionY,
        gridResolutionZ,
    }: {
        device: GPUDevice,
        gridResolutionX: number,
        gridResolutionY: number,
        gridResolutionZ: number,
    }) {
        this.device = device;
        this.simulationGridDims = [gridResolutionX, gridResolutionY, gridResolutionZ];
        
        const maxSimRes = Math.max(gridResolutionX, gridResolutionY, gridResolutionZ);
        // this.downsampleFactor = Math.max(1, Math.ceil(maxSimRes / MAX_MC_GRID_RES));
        
        // const mcResX = Math.ceil(gridResolutionX / this.downsampleFactor);
        // const mcResY = Math.ceil(gridResolutionY / this.downsampleFactor);
        // const mcResZ = Math.ceil(gridResolutionZ / this.downsampleFactor);
        // this.marchingCubesGridDims = [mcResX, mcResY, mcResZ];

        this.gridResolution = [gridResolutionX, gridResolutionY, gridResolutionZ];
        
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
        
        // Density grid (u32 per cell for atomics)
        const numCells = this.gridResolution[0] * this.gridResolution[1] * this.gridResolution[2];
        this.densityGridBuffer = device.createBuffer({
            label: "MC density grid buffer",
            size: numCells * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        
        // Vertex density (one extra in each dimension for corners)
        const numVertices = (this.gridResolution[0] + 1) * (this.gridResolution[1] + 1) * (this.gridResolution[2] + 1);
        this.vertexDensityBuffer = device.createBuffer({
            label: "MC vertex density buffer",
            size: numVertices * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        
        // Vertex gradient (vec4f per vertex for alignment) for precomputed normals
        this.vertexGradientBuffer = device.createBuffer({
            label: "MC vertex gradient buffer",
            size: numVertices * 16, // vec4f = 16 bytes
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Block-based optimization
        const blocksX = Math.ceil(this.gridResolution[0] / 8);
        const blocksY = Math.ceil(this.gridResolution[1] / 8);
        const blocksZ = Math.ceil(this.gridResolution[2] / 8);
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
        return this.gridResolution[0] * this.gridResolution[1] * this.gridResolution[2];
    }
    
    get numVertices(): number {
        return (this.gridResolution[0] + 1) * (this.gridResolution[1] + 1) * (this.gridResolution[2] + 1);
    }
    
    get gridDims(): [number, number, number] {
        return this.gridResolution;
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
