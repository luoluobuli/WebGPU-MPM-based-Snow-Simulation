import commonModuleSrc from "./shader/_common.wgsl?raw";
import vertexModuleSrc from "./shader/vertex.wgsl?raw";
import fragmentModuleSrc from "./shader/fragment.wgsl?raw";

export const setupGpuPipelines = ({
    device,
    format,
    nParticles,
}: {
    device: GPUDevice,
    format: GPUTextureFormat,
    nParticles: number,
}) => {
    const particlePosBuffer = device.createBuffer({
        size: nParticles * 4 * 4,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    const particlePosArray = new Float32Array(nParticles * 4);
    for (let i = 0; i < nParticles; i++) {
        particlePosArray[i * 4] = Math.random();
        particlePosArray[i * 4 + 1] = Math.random();
        particlePosArray[i * 4 + 2] = Math.random();
        particlePosArray[i * 4 + 3] = 1;
    }
    device.queue.writeBuffer(particlePosBuffer, 0, particlePosArray);


    const uniformsBuffer = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    });

    const renderBindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
                buffer: {
                    type: "uniform",
                },
            },
        ],
    });
    const renderBindGroup = device.createBindGroup({
        layout: renderBindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: uniformsBuffer,
                },
            },
        ],
    });
    const renderPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [renderBindGroupLayout],
    });


    const vertexModule = device.createShaderModule({code: commonModuleSrc + vertexModuleSrc});
    const fragmentModule = device.createShaderModule({code: commonModuleSrc + fragmentModuleSrc});
    
    const renderPipeline = device.createRenderPipeline({
        vertex: {
            module: vertexModule,
            entryPoint: "vert",
            buffers: [
                {
                    attributes: [
                        {
                            shaderLocation: 0,
                            offset: 0,
                            format: "float32x4",
                        },
                    ],
                    arrayStride: 16,
                    stepMode: "vertex",
                },
            ],
        },

        fragment: {
            module: fragmentModule,
            entryPoint: "frag",
            targets: [
                {
                    format,
                },
            ],
        },

        primitive: {
            topology: "point-list",
        },

        layout: renderPipelineLayout,
    });

    return {particlePosBuffer, uniformsBuffer, renderBindGroup, renderPipeline};
};