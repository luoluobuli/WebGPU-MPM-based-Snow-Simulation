export const requestGpuDeviceAndContext = async ({
    onStatusChange,
    onErr,
    canvas,
}: {
    onStatusChange: (text: string) => void,
    onErr: (text: string) => void,
    canvas: HTMLCanvasElement,
}) => {
    onStatusChange("accessing gpu adapter");
    if (navigator.gpu === undefined) {
        onErr("webgpu not supported");
        return null;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (adapter === null) {
        onErr("could not get adapter");
        return null;
    }

    onStatusChange("accessing gpu device");
    const device = await adapter.requestDevice({
        requiredLimits: {

        },
    });
    if (device === null) {
        onErr("could not get device");
        return null;
    }

    device.lost.then(() => {
        onErr("gpu device was lost");
    });


    const context = canvas.getContext("webgpu");
    if (context === null) {
        onErr("could not get context");
        return null;
    }

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format,
        alphaMode: "premultiplied",
    });


    return {device, context, format};
};