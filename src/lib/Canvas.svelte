<script lang="ts">
import { onMount } from "svelte";
import { requestGpuDeviceAndContext } from "./gpu-device";
import headerModuleSrc from "./shader/_header.wgsl?raw";
import vertexModuleSrc from "./shader/vertex.wgsl?raw";
import fragmentModuleSrc from "./shader/fragment.wgsl?raw";

let {
    onStatusChange,
    onErr,
}: {
    onStatusChange: (text: string) => void,
    onErr: (text: string) => void,
} = $props();


let canvas: HTMLCanvasElement;


onMount(async () => {
    const response = await requestGpuDeviceAndContext({onStatusChange, onErr, canvas});
    if (response === null) return;

    const {device, context, format} = response;


    const N_PARTICLES = 2_000;

    const particlePos = new Float32Array(N_PARTICLES * 16);
    const particlePosBuffer = device.createBuffer({
        size: particlePos.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    for (let i = 0; i < N_PARTICLES; i++) {
        particlePos[i * 4] = Math.random();
        particlePos[i * 4 + 1] = Math.random();
        particlePos[i * 4 + 2] = Math.random();
        particlePos[i * 4 + 3] = 1;
    }
    device.queue.writeBuffer(particlePosBuffer, 0, particlePos);


    const bindGroupLayout = device.createBindGroupLayout({
        entries: [],
    });
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [],
    });
    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
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

        layout: pipelineLayout,
    });


    const commandEncoder = device.createCommandEncoder();

    const renderPassEncoder = commandEncoder.beginRenderPass({
        colorAttachments: [
            {
                clearValue: {
                    r: 0,
                    g: 0,
                    b: 0,
                    a: 1,
                },

                loadOp: "clear",
                storeOp: "store",
                view: context.getCurrentTexture().createView(),
            },
        ],
    });
    renderPassEncoder.setBindGroup(0, bindGroup);
    renderPassEncoder.setVertexBuffer(0, particlePosBuffer);
    renderPassEncoder.setPipeline(renderPipeline);
    renderPassEncoder.draw(N_PARTICLES);
    renderPassEncoder.end();


    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    onStatusChange("done!");
});
</script>


<canvas
    bind:this={canvas}
>

</canvas>