// Maximum triangles per cell is 5, reasonable estimate for max total
// WebGPU default maxBufferSize is 256MB (268,435,456 bytes).
// 3,500,000 tris * 3 verts * 24 bytes (packed) = 252,000,000 bytes.
// This fits within 256MB.
const MAX_TRIANGLES = 3500000;
const MAX_VERTICES = MAX_TRIANGLES * 3;

// Maximum MC grid resolution per axis (will downsample if larger)
const MAX_MC_GRID_RES = 256;

export class GpuMarchingCubesBufferManager {
    readonly device: GPUDevice;
    
    // Vertex buffer: each vertex has position (vec3f) + normal (vec3f) = 24 bytes
    readonly vertexBuffer: GPUBuffer;
    
    // Indirect draw buffer: vertexCount, instanceCount, firstVertex, firstInstance
    readonly indirectDrawBuffer: GPUBuffer;
    
    // Atomic counter for vertex allocation
    readonly atomicCounterBuffer: GPUBuffer;
    
    // Density grid (uses MPM's mass grid, but we need our own smoothed version)
    readonly densityGridBuffer: GPUBuffer;
    
    // Vertex density buffer (one value per grid vertex)
    readonly vertexDensityBuffer: GPUBuffer;
    
    // Vertex gradient buffer (for precomputed normals)
    readonly vertexGradientBuffer: GPUBuffer;

    // Active blocks list for sparse update
    readonly activeBlocksBuffer: GPUBuffer;
    readonly blockIndirectDispatchBuffer: GPUBuffer;
    readonly marchingCubesGridDims: [number, number, number];
    readonly simulationGridDims: [number, number, number];
    readonly downsampleFactor: number;
    
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
        this.downsampleFactor = Math.max(1, Math.ceil(maxSimRes / MAX_MC_GRID_RES));
        
        const mcResX = Math.ceil(gridResolutionX / this.downsampleFactor);
        const mcResY = Math.ceil(gridResolutionY / this.downsampleFactor);
        const mcResZ = Math.ceil(gridResolutionZ / this.downsampleFactor);
        this.marchingCubesGridDims = [mcResX, mcResY, mcResZ];
        
        this.vertexBuffer = device.createBuffer({
            label: "MC vertex buffer",
            size: MAX_VERTICES * 24, // Packed f32 arrays
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
        const numCells = mcResX * mcResY * mcResZ;
        this.densityGridBuffer = device.createBuffer({
            label: "MC density grid buffer",
            size: numCells * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        
        // Vertex density (one extra in each dimension for corners)
        const numVertices = (mcResX + 1) * (mcResY + 1) * (mcResZ + 1);
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
        const blocksX = Math.ceil(mcResX / 8);
        const blocksY = Math.ceil(mcResY / 8);
        const blocksZ = Math.ceil(mcResZ / 8);
        const totalBlocks = blocksX * blocksY * blocksZ;

        this.activeBlocksBuffer = device.createBuffer({
            label: "MC active blocks buffer",
            size: totalBlocks * 4, // u32 block index
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.blockIndirectDispatchBuffer = device.createBuffer({
            label: "MC block indirect dispatch buffer",
            size: 12, // x, y, z (atomic x)
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        });
    }
    
    get numCells(): number {
        return this.marchingCubesGridDims[0] * this.marchingCubesGridDims[1] * this.marchingCubesGridDims[2];
    }
    
    get numVertices(): number {
        return (this.marchingCubesGridDims[0] + 1) * (this.marchingCubesGridDims[1] + 1) * (this.marchingCubesGridDims[2] + 1);
    }
    
    get gridDims(): [number, number, number] {
        return this.marchingCubesGridDims;
    }
}
