@group(1) @binding(0) var<storage, read> particleData: array<ParticleData>;

const SPHERE_RADIUS = 0.05;
const SMOOTHNESS = 1 / f32(2 << 4);

fn smoothMin(a: f32, b: f32) -> f32 {
    // quartic polynomial smooth min from https://iquilezles.org/articles/smin/
    let smoothFac = SMOOTHNESS * 16 / 3;
    let blendFac = max(0, smoothFac - abs(a - b)) / smoothFac;
    return min(a, b) - blendFac * blendFac * blendFac * (4 - blendFac) * smoothFac / 16;
}

fn allParticlesSdf(pos: vec3f) -> f32 {
    var dist = length(pos - particleData[0].pos) - SPHERE_RADIUS;
    
    for (var i = 1u; i < arrayLength(&particleData); i++) {
        let curDist = length(pos - particleData[i].pos) - SPHERE_RADIUS;
        dist = smoothMin(dist, curDist);
    }
    
    return dist;
}

fn calcNormal(pos: vec3f) -> vec3f {
    const EPSILON = 1e-4;

    // normal is the direction of the gradient/derivative
    return normalize(vec3f(
        allParticlesSdf(pos + vec3f(EPSILON, 0, 0)) - allParticlesSdf(pos - vec3f(EPSILON, 0, 0)),
        allParticlesSdf(pos + vec3f(0, EPSILON, 0)) - allParticlesSdf(pos - vec3f(0, EPSILON, 0)),
        allParticlesSdf(pos + vec3f(0, 0, EPSILON)) - allParticlesSdf(pos - vec3f(0, 0, EPSILON)),
    ));
}

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
    const MAX_N_STEPS = 100u;
    const COLLISION_DIST = 1e-4;
    const MAX_DIST = 100;

    for (var nStep = 0u; nStep < MAX_N_STEPS; nStep++) {
        let dist = allParticlesSdf(rayPos);
        rayPos += rayDir * dist;

        if dist < COLLISION_DIST {
            found = true;
            break;
        }

        if dist > MAX_DIST { break; }
    }

    if !found { return vec4f(0, 0, 0, 1); }
    
    let normal = calcNormal(rayPos);
    return vec4f(normal * 0.5 + 0.5, 1);
}