@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> particles: array<ParticleData>;
@group(1) @binding(1) var<storage, read> spawnPoints: array<vec4f>;

@compute
@workgroup_size(256)
fn scatterParticles(
    @builtin(global_invocation_id) globalId: vec3u,
) {
    let threadIndex = globalId.x;
    if threadIndex >= arrayLength(&particles) { return; }

    let spawnPointIndex = min(threadIndex, arrayLength(&spawnPoints) - 1u);
    let candidatePos = clamp(
        spawnPoints[spawnPointIndex].xyz,
        uniforms.gridMinCoords + uniforms.gridCellDims,
        uniforms.gridMaxCoords - uniforms.gridCellDims,
    );

    let particle = &particles[threadIndex];

    (*particle).pos = candidatePos;
    (*particle)._hom = 1;
    (*particle).vel = vec3f();
    (*particle).mass = 1.0 / 3.0;
    (*particle).deformationElastic = IDENTITY_MAT3;
    (*particle).deformationPlastic = IDENTITY_MAT3;

    (*particle).pos_displacement = vec3f();
    (*particle).deformation_displacement = mat3x3f();
}
