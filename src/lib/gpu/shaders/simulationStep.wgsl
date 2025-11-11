@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read> particleDataIn: array<ParticleData>;
@group(1) @binding(1) var<storage, read_write> particleDataOut: array<ParticleData>;

@compute
@workgroup_size(256)
fn doSimulationStep(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let threadIndex = gid.x;
    if threadIndex > arrayLength(&particleDataIn) { return; }

    var particle = particleDataIn[threadIndex];

    particle.vel += vec3f(0, 0, -9.81 * uniforms.simulationTimestep);
    particle.pos += particle.vel * uniforms.simulationTimestep;

    particleDataOut[threadIndex] = particle;
}