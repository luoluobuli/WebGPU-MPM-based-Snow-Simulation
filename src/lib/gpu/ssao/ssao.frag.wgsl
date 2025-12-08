
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

@fragment
fn frag(input: FragmentInput) -> @location(0) vec4f {
    let dim = textureDimensions(depthTexture);
    let coords = vec2i(input.uv * vec2f(dim));
    
    if (coords.x < 0 || coords.x >= i32(dim.x) || coords.y < 0 || coords.y >= i32(dim.y)) {
        return vec4f(1.0);
    }

    let depth = textureLoad(depthTexture, coords, 0);

    if depth >= 1 - 1e-4 {
         // No occlusion on bg
        return vec4f(0);
    }

    let position = unproject(input.uv, depth);
    let normal = normalize(cross(dpdx(position), dpdy(position)));
    
    const SSAO_RADIUS = 0.5;
    const SSAO_BIAS = 0.15;
    const N_SSAO_SAMPLES = 16u;


    var occlusion = 0.;
    for (var i = 0u; i < N_SSAO_SAMPLES; i++) {
        let rand_vector = vec3f(
            f32(hash3(bitcast<vec3u>(vec3f(input.uv, f32(i)) + position * 10))),
            f32(hash3(bitcast<vec3u>(vec3f(input.uv, f32(i + N_SSAO_SAMPLES * 2)) + position * 10))),
            f32(hash3(bitcast<vec3u>(vec3f(input.uv, f32(i + N_SSAO_SAMPLES * 4)) + position * 10))),
        ) / f32(0xFFFFFFFF);
        var tangentSample = normalize(rand_vector * 2 - 1);
        
        tangentSample *= sign(dot(tangentSample, normal));
        
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

            let occluderPos = unproject(offsetUV, sampleDepthVal);
            
            let distSampleToCam = distance(uniforms.cameraPos, sampleWorldPos);
            let distOccluderToCam = distance(uniforms.cameraPos, occluderPos);
            
            let distToOrigin = distance(position, occluderPos);
            let rangeCheck = smoothstep(0, 1, SSAO_RADIUS / (distToOrigin + 0.001));
            
            occlusion += select(0, rangeCheck, distOccluderToCam < distSampleToCam - SSAO_BIAS);
        }
    }
    
    let finalOcc = pow(occlusion / f32(N_SSAO_SAMPLES), 2);
    
    return vec4f(0.1, 0.2, 0.3, 1) * finalOcc;
}
