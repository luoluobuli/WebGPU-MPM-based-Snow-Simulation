// Maximum triangles per cell is 5, reasonable estimate for max total
const MAX_TRIANGLES = 500000;
const MAX_VERTICES = MAX_TRIANGLES * 3;

// Maximum MC grid resolution per axis (will downsample if larger)
const MAX_MC_GRID_RES = 64;

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
    
    // Actual MC grid dimensions (may be downsampled from simulation grid)
    readonly mcGridDims: [number, number, number];
    
    // Original simulation grid dimensions
    readonly simGridDims: [number, number, number];
    
    // Downsample factor
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
        this.simGridDims = [gridResolutionX, gridResolutionY, gridResolutionZ];
        
        // Calculate downsample factor to keep resolution reasonable
        const maxSimRes = Math.max(gridResolutionX, gridResolutionY, gridResolutionZ);
        this.downsampleFactor = Math.max(1, Math.ceil(maxSimRes / MAX_MC_GRID_RES));
        
        const mcResX = Math.ceil(gridResolutionX / this.downsampleFactor);
        const mcResY = Math.ceil(gridResolutionY / this.downsampleFactor);
        const mcResZ = Math.ceil(gridResolutionZ / this.downsampleFactor);
        this.mcGridDims = [mcResX, mcResY, mcResZ];
        
        console.log(`MC grid: ${mcResX}x${mcResY}x${mcResZ} (downsampled ${this.downsampleFactor}x from simulation grid)`);
        
        // Vertex buffer: position (aligned vec3f = 16 bytes) + normal (aligned vec3f = 16 bytes) = 32 bytes per vertex
        this.vertexBuffer = device.createBuffer({
            label: "MC vertex buffer",
            size: MAX_VERTICES * 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        
        // Indirect draw buffer for drawIndirect
        // Format: vertexCount (u32), instanceCount (u32), firstVertex (u32), firstInstance (u32)
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
    }
    
    get numCells(): number {
        return this.mcGridDims[0] * this.mcGridDims[1] * this.mcGridDims[2];
    }
    
    get numVertices(): number {
        return (this.mcGridDims[0] + 1) * (this.mcGridDims[1] + 1) * (this.mcGridDims[2] + 1);
    }
    
    get gridDims(): [number, number, number] {
        return this.mcGridDims;
    }
}
