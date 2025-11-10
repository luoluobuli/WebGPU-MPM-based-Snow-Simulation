import headerModuleSrc from "./shader/_header.wgsl?raw";
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
    const particlePos = new Float32Array(nParticles * 4);
    const particlePosBuffer = device.createBuffer({
        size: particlePos.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    for (let i = 0; i < nParticles; i++) {
        particlePos[i * 4] = Math.random();
        particlePos[i * 4 + 1] = Math.random();
        particlePos[i * 4 + 2] = Math.random();
        particlePos[i * 4 + 3] = 1;
    }
    device.queue.writeBuffer(particlePosBuffer, 0, particlePos);


    const renderBindGroupLayout = device.createBindGroupLayout({
        entries: [],
    });
    const renderBindGroup = device.createBindGroup({
        layout: renderBindGroupLayout,
        entries: [],
    });
    const renderPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [renderBindGroupLayout],
    });


    const vertexModule = device.createShaderModule({code: headerModuleSrc + vertexModuleSrc});
    const fragmentModule = device.createShaderModule({code: headerModuleSrc + fragmentModuleSrc});
    
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

    return {particlePosBuffer, renderBindGroup, renderPipeline};
};