export class ElapsedTime {
    gpuComputeSimulationStepTimeNs = $state(0n);
    gpuComputeSimulationSubstepTimeNs = $state(0n);
    nSimulationSubsteps = $state(0);
    gpuRenderTimeNs = $state(0n);
    gpuPostprocessRenderTimeNs = $state(0n);


    animationFrameTimeNs = $state(0n);
}
