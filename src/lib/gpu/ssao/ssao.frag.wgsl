
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var depthTexture: texture_depth_2d;

struct FragmentInput {
    @location(0) uv: vec2f,
}

fn unproject(uv: vec2f, depth: f32) -> vec3f {
    let clipPos = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let viewPosH = uniforms.viewProjInvMat * clipPos;
    return viewPosH.xyz / viewPosH.w;
}

fn project(worldPos: vec3f) -> vec4f {
    return uniforms.viewProjMat * vec4f(worldPos, 1.0);
}

fn random01(coords: vec2i, index: u32, salt: u32) -> f32 {
    return f32(hash3(vec3u(u32(coords.x), u32(coords.y), index ^ salt))) / f32(0xffffffffu);
}

@fragment
fn frag(input: FragmentInput) -> @location(0) vec4f {
    let dim = textureDimensions(depthTexture);
    let maxCoords = vec2i(dim) - vec2i(1);
    let coords = clamp(vec2i(input.uv * vec2f(dim)), vec2i(0), maxCoords);

    let depth = textureLoad(depthTexture, coords, 0);
    let center_uv = (vec2f(coords) + 0.5) / vec2f(dim);
    let position = unproject(center_uv, depth);
    let view_dir = normalize(uniforms.cameraPos - position);
    let raw_normal = cross(dpdx(position), dpdy(position));
    let raw_normal_len = length(raw_normal);
    var normal = select(vec3f(0, 0, 1), raw_normal / raw_normal_len, raw_normal_len > 1e-5);
    normal *= select(-1.0, 1.0, dot(normal, view_dir) >= 0.0);
    let world_units_per_pixel = max(length(dpdx(position)), length(dpdy(position)));

    if depth >= 1 - 1e-4 {
         // No occlusion on bg
        return vec4f(0);
    }
    
    const SSAO_RADIUS = 0.65;
    const SSAO_BIAS = 0.035;
    const SSAO_STRENGTH = 2.2;
    const MIN_EFFECTIVE_SAMPLE_RADIUS_PX = 0.75;
    const FULL_EFFECTIVE_SAMPLE_RADIUS_PX = 4.0;
    const N_SSAO_SAMPLES = 16u;

    let projected_radius_px = SSAO_RADIUS / max(world_units_per_pixel, 1e-4);
    let projected_radius_fade = mix(0.35, 1.0, smoothstep(
        MIN_EFFECTIVE_SAMPLE_RADIUS_PX,
        FULL_EFFECTIVE_SAMPLE_RADIUS_PX,
        projected_radius_px,
    ));
    let depth_bias = max(SSAO_BIAS, world_units_per_pixel * 0.06);

    var occlusion = 0.;
    for (var i = 0u; i < N_SSAO_SAMPLES; i++) {
        let rand_vector = vec3f(
            random01(coords, i, 0x9e3779b9u),
            random01(coords, i, 0x85ebca6bu),
            random01(coords, i, 0xc2b2ae35u),
        );
        var tangentSample = normalize(rand_vector * 2 - 1);
        
        tangentSample *= select(-1.0, 1.0, dot(tangentSample, normal) >= 0.0);
        
        // Sample distribution (concentrate near center)
        let scale = f32(i) / f32(N_SSAO_SAMPLES);
        let dist = SSAO_RADIUS * mix(0.1, 1, scale * scale * scale);
        
        let sampleWorldPos = position + tangentSample * dist;
        
        let offsetClip = project(sampleWorldPos);
        let offsetNDC = offsetClip.xyz / offsetClip.w;
        let offsetUV = vec2f(offsetNDC.x * 0.5 + 0.5, 1.0 - (offsetNDC.y * 0.5 + 0.5));
        
        if (offsetUV.x >= 0.0 && offsetUV.x <= 1.0 && offsetUV.y >= 0.0 && offsetUV.y <= 1.0) {
            let sampleCoords = vec2i(offsetUV * vec2f(dim));
            let sampleDepthVal = textureLoad(depthTexture, sampleCoords, 0);
            if sampleDepthVal >= 1 - 1e-4 {
                continue;
            }

            let occluderPos = unproject(offsetUV, sampleDepthVal);
            let sample_ray = normalize(sampleWorldPos - uniforms.cameraPos);
            let sample_ray_distance = dot(sampleWorldPos - uniforms.cameraPos, sample_ray);
            let occluder_ray_distance = dot(occluderPos - uniforms.cameraPos, sample_ray);
            let depth_delta = sample_ray_distance - occluder_ray_distance;
            
            let distToOrigin = distance(position, occluderPos);
            let rangeCheck = 1.0 - smoothstep(SSAO_RADIUS * 0.35, SSAO_RADIUS * 1.15, distToOrigin);
            let occluderWeight = smoothstep(depth_bias, depth_bias + SSAO_RADIUS * 0.18, depth_delta);
            
            occlusion += rangeCheck * occluderWeight;
        }
    }
    
    let finalOcc = saturate(pow(occlusion / f32(N_SSAO_SAMPLES), 1.15) * SSAO_STRENGTH * projected_radius_fade);
    
    return vec4f(0.1, 0.2, 0.3, 1) * finalOcc;
}
