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

fn rayIntersectsTriangle(
    rayOrigin: vec3f,
    rayDir: vec3f,
    v0: vec3f,
    v1: vec3f,
    v2: vec3f,
) -> bool {
    let EPSILON = 1e-6;
    
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(rayDir, edge2);
    let a = dot(edge1, h);
    
    // parallel chcek
    if abs(a) < EPSILON { return false; }
    
    let f = 1 / a;
    let s = rayOrigin - v0;
    let u = f * dot(s, h);
    
    if u < 0 || u > 1 { return false; }
    
    let q = cross(s, edge1);
    let v = f * dot(rayDir, q);
    
    if v < 0 || u + v > 1 { return false; }
    
    let t = f * dot(edge2, q);
    
    return t > EPSILON;
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
    

    particles[threadIndex].pos = candidatePos;
    particles[threadIndex]._hom = 1;
    particles[threadIndex].vel = vec3f(0);
    particles[threadIndex].affine = vec3f(0);
    particles[threadIndex].mass = 1;
    particles[threadIndex].deformation = mat3x3f(
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    );
}
