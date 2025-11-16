@group(1) @binding(0) var<storage, read> particleData: array<ParticleData>;

@fragment
fn frag(
    in: RaymarchVertexOut,
) -> @location(0) vec4f {
    if arrayLength(&particleData) == 0 { return vec4f(0, 0, 0, 1); }

    // use inverse view-projection to get ray direction that respects FOV and aspect ratio
    let nearPointHom = uniforms.viewProjInvMat * vec4f(in.uvCentered, 0, 1);
    let farPointHom = uniforms.viewProjInvMat * vec4f(in.uvCentered, 1, 1);
    
    let nearPoint = nearPointHom.xyz / nearPointHom.w;
    let farPoint = farPointHom.xyz / farPointHom.w;
    


    let rayDir = normalize(farPoint - nearPoint);
    var rayPos = nearPoint;


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