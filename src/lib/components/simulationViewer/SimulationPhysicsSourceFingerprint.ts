import runnerSource from "../../gpu/GpuSnowPipelineRunner.svelte.ts?raw";
import simulationTimestepSource from "../../gpu/simulationTimestep.ts?raw";
import mpmPipelineSource from "../../gpu/mpm/GpuMpmPipelineManager.ts?raw";
import particleToGridSource from "../../gpu/mpm/particleToGrid.cs.wgsl?raw";
import clearGridAccumulatorSource from "../../gpu/mpm/clearGridAccumulator.wgsl?raw";
import gridUpdateSource from "../../gpu/mpm/gridUpdate.cs.wgsl?raw";
import gridToParticleSource from "../../gpu/mpm/gridToParticle.cs.wgsl?raw";
import integrateParticlesSource from "../../gpu/mpm/integrateParticles.wgsl?raw";
import fusedMlsSource from "../../gpu/mpm/fusedMlsG2p2g.cs.wgsl?raw";
import sparseGridSource from "../../gpu/mpm/sparseGridPrelude.wgsl?raw";
import particleScatterSource from "../../gpu/particleInitialize/particleScatter.cs.wgsl?raw";
import constantsSource from "../../gpu/shaderPrelude/constants.wgsl?raw";
import uniformsSource from "../../gpu/shaderPrelude/uniforms.wgsl?raw";
import randomSource from "../../gpu/shaderPrelude/randomNumberGenerators.wgsl?raw";
import matrixOpsSource from "../../gpu/shaderPrelude/matrixOps.wgsl?raw";
import mpmOpsSource from "../../gpu/shaderPrelude/mpmOps.wgsl?raw";
import stressTensorSource from "../../gpu/shaderPrelude/stressTensorOps.wgsl?raw";
import plasticitySource from "../../gpu/shaderPrelude/plasticityOps.wgsl?raw";
import rigidPawManagerSource from "../../gpu/kinematicCollider/GpuRigidPawSdfKinematicColliderManager.ts?raw";
import kinematicIntervalBatchSource from "../../gpu/kinematicCollider/GpuKinematicColliderIntervalBatchBuffer.ts?raw";
import rigidPawColliderSource from "../../gpu/kinematicCollider/rigidPawSdfKinematicColliderPrelude.wgsl?raw";
import rigidPawAssetSource from "../../gpu/collider/rigidPawSdfColliderAsset.ts?raw";
import zubeiaGroundSource from "./zubeia/ZubeiaGround.ts?raw";
import zubeiaWalkCalibrationSource from "./zubeia/ZubeiaWalkCalibration.ts?raw";
import zubeiaPawCalibrationSource from "./zubeia/ZubeiaPawColliderCalibration.ts?raw";
import zubeiaPawLoaderSource from "./zubeia/loadZubeiaRigidPawWalkCollider.ts?raw";
import { fingerprintString } from "./SimulationCacheFingerprint";

// Cached frames contain solver output, so source changes that can alter that
// output must create a different cache namespace. This is separate from the
// packed-frame schema because identical bytes can encode different physics.
export const SIMULATION_FRAME_CACHE_SCHEMA_REVISION =
    "position-material-v2";

export const SIMULATION_PHYSICS_SOURCE_FINGERPRINT = fingerprintString([
    runnerSource,
    simulationTimestepSource,
    mpmPipelineSource,
    particleToGridSource,
    gridUpdateSource,
    clearGridAccumulatorSource,
    gridToParticleSource,
    integrateParticlesSource,
    fusedMlsSource,
    sparseGridSource,
    particleScatterSource,
    constantsSource,
    uniformsSource,
    randomSource,
    matrixOpsSource,
    mpmOpsSource,
    stressTensorSource,
    plasticitySource,
    rigidPawManagerSource,
    kinematicIntervalBatchSource,
    rigidPawColliderSource,
    rigidPawAssetSource,
    zubeiaGroundSource,
    zubeiaWalkCalibrationSource,
    zubeiaPawCalibrationSource,
    zubeiaPawLoaderSource,
].join("\0"));
