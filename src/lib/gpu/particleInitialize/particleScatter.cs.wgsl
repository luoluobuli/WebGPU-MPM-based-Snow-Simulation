@group(1) @binding(0) var<storage, read_write> particles: array<ParticleData>;
@group(1) @binding(1) var<storage, read> meshVertices: array<vec3f>;


fn randFloat(seed: ptr<function, u32>) -> f32 {
    *seed = hash1(*seed);
    return f32(*seed) / f32(0xFFFFFFFF);
}

fn randVec3(seed: ptr<function, u32>, minCoords: vec3f, maxCoords: vec3f) -> vec3f {
    return vec3f(
        mix(minCoords.x, maxCoords.x, randFloat(seed)),
        mix(minCoords.y, maxCoords.y, randFloat(seed)),
        mix(minCoords.z, maxCoords.z, randFloat(seed)),
    );
}

fn pointInsideMesh(point: vec3f, numTriangles: u32) -> bool {
    // even-odd check in +x direction

    let rayDir = vec3f(1, 0, 0);
    var inside = false;
    
    for (var i = 0u; i < numTriangles; i++) {
        let vert0 = meshVertices[i * 3];
        let vert1 = meshVertices[i * 3 + 1];
        let vert2 = meshVertices[i * 3 + 2];
        
        if rayIntersectsTriangle(point, rayDir, vert0, vert1, vert2) {
            inside = !inside;
        }
    }
    
    return inside;
}

@compute
@workgroup_size(256)
fn scatterParticles(
    @builtin(global_invocation_id) globalId: vec3u,
) {
    let threadIndex = globalId.x;
    if threadIndex >= arrayLength(&particles) { return; }



    const REJECTION_SAMPLING_N_MAX_ATTEMPTS = 128u;

    let nTriangles = arrayLength(&meshVertices) / 3;
    var seed = hash4(vec4u(threadIndex, 100, 200, 300));
    var candidatePos = vec3f(0);

    for (var nAttempt = 0u; nAttempt < REJECTION_SAMPLING_N_MAX_ATTEMPTS; nAttempt++) {
        candidatePos = randVec3(&seed, uniforms.meshMinCoords, uniforms.meshMaxCoords);
        if pointInsideMesh(candidatePos, nTriangles) { break; }
    }

    let noiseScale = 0.1;
    let NOISE_FREQ = 4.;
    let dx = fbm(candidatePos * NOISE_FREQ);
    let dy = fbm(candidatePos * NOISE_FREQ + vec3f(100, 200, 300));
    let dz = fbm(candidatePos * NOISE_FREQ + vec3f(-300, -200, -100));
    candidatePos += (vec3f(dx, dy, dz) - 0.5) * noiseScale;


    let particle = &particles[threadIndex];
    

    let mass1 = f32(hash1(threadIndex)) / f32(0xFFFFFFFF);

    (*particle).pos = candidatePos;
    (*particle)._hom = 1;
    (*particle).vel = vec3f();
    (*particle).mass = mass1 * mass1;
    (*particle).deformationElastic = IDENTITY_MAT3;
    (*particle).deformationPlastic = IDENTITY_MAT3;

    (*particle).pos_displacement = vec3f();
    (*particle).deformation_displacement = mat3x3f(); // Zero matrix - represents change in deformation
}
