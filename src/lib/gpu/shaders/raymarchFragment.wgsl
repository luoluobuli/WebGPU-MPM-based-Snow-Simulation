@group(1) @binding(0) var<storage, read> particleData: array<ParticleData>;

@fragment
fn frag(
    in: RaymarchVertexOut,
) -> @location(0) vec4f {
    if arrayLength(&particleData) == 0 { return vec4f(0, 0, 0, 1); }

    let rayDir = normalize((uniforms.viewInvMat * vec4f(vec3f(in.uvCentered, -1), 0)).xyz);
    
    var rayPos = (uniforms.viewInvMat * vec4f(0, 0, 0, 1)).xyz;


    var found = false;
    var particleHitPos: vec3f;

    const SPHERE_RADIUS = 0.05;
    const MAX_N_STEPS = 20u;

    for (var nStep = 0u; nStep < MAX_N_STEPS; nStep++) {
        var closestParticleIndex = 0u;

        var closestParticleDist = length(rayPos - particleData[0].pos) - SPHERE_RADIUS;
        for (var i = 1u; i < arrayLength(&particleData); i++) {
            let dist = length(rayPos - particleData[i].pos) - SPHERE_RADIUS;

            if dist < closestParticleDist {
                closestParticleIndex = i;
                closestParticleDist = dist;
            }
        }

        rayPos += rayDir * closestParticleDist;

        if closestParticleDist < 1e-3 {
            found = true;
            particleHitPos = particleData[closestParticleIndex].pos;
            break;
        }
    }


    if !found { return vec4f(0, 0, 0, 1); }
    return vec4f(normalize(rayPos - particleHitPos).xyz * 0.5 + 0.5, 1);
}