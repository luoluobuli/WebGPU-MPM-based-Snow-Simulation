
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var depthTexture: texture_depth_2d;

const BACKGROUND_DEPTH = 1.0 - 1e-4;

struct FragmentInput {
    @location(0) uv: vec2f,
}

struct SurfaceInfo {
    normal: vec3f,
    worldUnitsPerPixel: f32,
}

fn pixelUv(coords: vec2i, dim: vec2u) -> vec2f {
    return (vec2f(coords) + 0.5) / vec2f(dim);
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

fn reconstructSurface(coords: vec2i, depth: f32, position: vec3f, view_dir: vec3f, dim: vec2u) -> SurfaceInfo {
    let maxCoords = vec2i(dim) - vec2i(1);
    let leftCoords = clamp(coords + vec2i(-1, 0), vec2i(0), maxCoords);
    let rightCoords = clamp(coords + vec2i(1, 0), vec2i(0), maxCoords);
    let upCoords = clamp(coords + vec2i(0, -1), vec2i(0), maxCoords);
    let downCoords = clamp(coords + vec2i(0, 1), vec2i(0), maxCoords);

    let depthLeft = textureLoad(depthTexture, leftCoords, 0);
    let depthRight = textureLoad(depthTexture, rightCoords, 0);
    let depthUp = textureLoad(depthTexture, upCoords, 0);
    let depthDown = textureLoad(depthTexture, downCoords, 0);

    let validLeft = coords.x > 0 && depthLeft < BACKGROUND_DEPTH;
    let validRight = coords.x < maxCoords.x && depthRight < BACKGROUND_DEPTH;
    let validUp = coords.y > 0 && depthUp < BACKGROUND_DEPTH;
    let validDown = coords.y < maxCoords.y && depthDown < BACKGROUND_DEPTH;

    let posLeft = unproject(pixelUv(leftCoords, dim), depthLeft);
    let posRight = unproject(pixelUv(rightCoords, dim), depthRight);
    let posUp = unproject(pixelUv(upCoords, dim), depthUp);
    let posDown = unproject(pixelUv(downCoords, dim), depthDown);

    var dPdx = vec3f();
    var hasDx = false;
    if validLeft && validRight {
        let leftDelta = abs(depth - depthLeft);
        let rightDelta = abs(depth - depthRight);
        dPdx = select(position - posLeft, posRight - position, rightDelta < leftDelta);
        hasDx = true;
    } else if validLeft {
        dPdx = position - posLeft;
        hasDx = true;
    } else if validRight {
        dPdx = posRight - position;
        hasDx = true;
    }

    var dPdy = vec3f();
    var hasDy = false;
    if validUp && validDown {
        let upDelta = abs(depth - depthUp);
        let downDelta = abs(depth - depthDown);
        dPdy = select(position - posUp, posDown - position, downDelta < upDelta);
        hasDy = true;
    } else if validUp {
        dPdy = position - posUp;
        hasDy = true;
    } else if validDown {
        dPdy = posDown - position;
        hasDy = true;
    }

    let rawNormal = cross(dPdx, dPdy);
    let rawNormalLen = length(rawNormal);

    var normal = view_dir;
    if hasDx && hasDy && rawNormalLen > 1e-5 {
        normal = rawNormal / rawNormalLen;
        normal *= select(-1.0, 1.0, dot(normal, view_dir) >= 0.0);
    }

    var worldUnitsPerPixel = max(length(dPdx), length(dPdy));
    if !hasDx && hasDy {
        worldUnitsPerPixel = length(dPdy);
    } else if hasDx && !hasDy {
        worldUnitsPerPixel = length(dPdx);
    } else if !hasDx && !hasDy {
        worldUnitsPerPixel = 1e6;
    }

    return SurfaceInfo(normal, max(worldUnitsPerPixel, 1e-4));
}

@fragment
fn frag(input: FragmentInput) -> @location(0) vec4f {
    let dim = textureDimensions(depthTexture);
    let maxCoords = vec2i(dim) - vec2i(1);
    let coords = clamp(vec2i(input.uv * vec2f(dim)), vec2i(0), maxCoords);

    let depth = textureLoad(depthTexture, coords, 0);
    let center_uv = pixelUv(coords, dim);

    if depth >= BACKGROUND_DEPTH {
         // No occlusion on bg
        return vec4f(0);
    }

    let position = unproject(center_uv, depth);
    let view_dir = normalize(uniforms.cameraPos - position);
    let surface = reconstructSurface(coords, depth, position, view_dir, dim);
    let normal = surface.normal;
    let normal_view = saturate(dot(normal, view_dir));
    let world_units_per_pixel = surface.worldUnitsPerPixel;
    
    const SSAO_RADIUS = 0.65;
    const SSAO_BIAS = 0.035;
    const SSAO_STRENGTH = 2.2;
    const MIN_EFFECTIVE_SAMPLE_RADIUS_PX = 1.25;
    const FULL_EFFECTIVE_SAMPLE_RADIUS_PX = 4.0;
    const N_SSAO_SAMPLES = 16u;

    let projected_radius_px = SSAO_RADIUS / max(world_units_per_pixel, 1e-4);
    let projected_radius_fade = smoothstep(
        MIN_EFFECTIVE_SAMPLE_RADIUS_PX,
        FULL_EFFECTIVE_SAMPLE_RADIUS_PX,
        projected_radius_px,
    );
    let grazing_fade = smoothstep(0.08, 0.28, normal_view);
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
            let sampleCoords = clamp(vec2i(offsetUV * vec2f(dim)), vec2i(0), maxCoords);
            let sampleDepthVal = textureLoad(depthTexture, sampleCoords, 0);
            if sampleDepthVal >= BACKGROUND_DEPTH {
                continue;
            }

            let occluderPos = unproject(offsetUV, sampleDepthVal);
            let sample_ray = normalize(sampleWorldPos - uniforms.cameraPos);
            let sample_ray_distance = dot(sampleWorldPos - uniforms.cameraPos, sample_ray);
            let occluder_ray_distance = dot(occluderPos - uniforms.cameraPos, sample_ray);
            let depth_delta = sample_ray_distance - occluder_ray_distance;
            
            let occluder_offset = occluderPos - position;
            let distToOrigin = length(occluder_offset);
            let rangeCheck = 1.0 - smoothstep(SSAO_RADIUS * 0.35, SSAO_RADIUS * 1.15, distToOrigin);
            let occluderWeight = smoothstep(depth_bias, depth_bias + SSAO_RADIUS * 0.18, depth_delta);
            let normalSeparation = dot(occluder_offset, normal);
            let surfaceSeparationWeight = smoothstep(depth_bias * 0.5, max(depth_bias * 1.5, SSAO_RADIUS * 0.08), normalSeparation);
            
            occlusion += rangeCheck * occluderWeight * surfaceSeparationWeight;
        }
    }
    
    let finalOcc = saturate(pow(occlusion / f32(N_SSAO_SAMPLES), 1.15) * SSAO_STRENGTH * projected_radius_fade * grazing_fade);
    
    return vec4f(0.0, 0.0, 0.0, 1.0) * finalOcc;
}
