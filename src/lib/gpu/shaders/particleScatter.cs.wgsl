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
    vert0: vec3f,
    vert1: vec3f,
    vert2: vec3f,
) -> bool {
    // Möller-Trumbore

    let EPSILON = 1e-6;
    
    let edge1 = vert1 - vert0;
    let edge2 = vert2 - vert0;
    
    let rayDirCrossEdge2 = cross(rayDir, edge2);
    let det = dot(edge1, rayDirCrossEdge2);
    
    // ray nearly parallel to triangle
    if abs(det) < EPSILON { return false; }
    
    let detInv = 1 / det;
    let originToVert0 = rayOrigin - vert0;


    let bary0 = dot(originToVert0, rayDirCrossEdge2) * detInv;
    if bary0 < 0 || 1 < bary0 { return false; }
    
    let originToVert0CrossEdge1 = cross(originToVert0, edge1);
    let bary1 = dot(rayDir, originToVert0CrossEdge1) * detInv;
    
    if bary1 < 0 || 1 < bary1 || bary0 + bary1 > 1 { return false; }
    
    let intersectionDist = dot(edge2, originToVert0CrossEdge1) * detInv;
    return intersectionDist > EPSILON;
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


    let particle = &particles[threadIndex];
    

    (*particle).pos = candidatePos;
    (*particle)._hom = 1;
    (*particle).vel = vec3f(5, 0, 5);
    (*particle).mass = 10;
    (*particle).deformationElastic = mat3x3f(
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    );
    (*particle).deformationPlastic = mat3x3f(
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    );
}
