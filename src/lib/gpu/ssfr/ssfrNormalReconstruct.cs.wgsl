@group(0) @binding(1) var smoothedDepthTexture: texture_2d<f32>;
@group(0) @binding(2) var normalTexture: texture_storage_2d<rgba16float, write>;

fn reconstructWorldPos(coords: vec2i, depth: f32, screen_size: vec2f) -> vec3f {
    let uv = (vec2f(coords) + 0.5) / screen_size;
    let ndc = vec2f(uv.x, 1 - uv.y) * 2 - 1;
    
    let clip_pos = vec4f(ndc, depth, 1);
    
    let world_pos_hom = uniforms.viewProjInvMat * clip_pos;
    return world_pos_hom.xyz / world_pos_hom.w;
}

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let screen_size = vec2f(textureDimensions(smoothedDepthTexture));
    let coords = vec2i(global_id.xy);
    
    if any(coords >= vec2i(screen_size)) { return; }
    
    let center_data = textureLoad(smoothedDepthTexture, coords, 0);
    let center_depth = center_data.r;
    let compression = center_data.g;
    
    if center_depth >= 1 {
        // bg pixel
        textureStore(normalTexture, coords, vec4f(0, 0, 0, 0));
        return;
    }
    
    let depth_left = textureLoad(smoothedDepthTexture, coords + vec2i(-1, 0), 0).r;
    let depth_right = textureLoad(smoothedDepthTexture, coords + vec2i(1, 0), 0).r;
    let depth_up = textureLoad(smoothedDepthTexture, coords + vec2i(0, -1), 0).r;
    let depth_down = textureLoad(smoothedDepthTexture, coords + vec2i(0, 1), 0).r;
    
    let pos_center = reconstructWorldPos(coords, center_depth, screen_size);
    let pos_left = reconstructWorldPos(coords + vec2i(-1, 0), depth_left, screen_size);
    let pos_right = reconstructWorldPos(coords + vec2i(1, 0), depth_right, screen_size);
    let pos_up = reconstructWorldPos(coords + vec2i(0, -1), depth_up, screen_size);
    let pos_down = reconstructWorldPos(coords + vec2i(0, 1), depth_down, screen_size);
    
    // partial derivatives
    let dPdx = select(
        select(
            select(
                (pos_right - pos_center) * 0.5,
                pos_center - pos_left,
                depth_right >= 1,
            ),
            pos_right - pos_center,
            depth_left >= 1,
        ),
        vec3f(),
        depth_left >= 1 || depth_right >= 1,
    );
    let dPdy = select(
        select(
            select(
                (pos_down - pos_center) * 0.5,
                pos_down - pos_center,
                depth_down >= 1,
            ),
            pos_down - pos_center,
            depth_up >= 1,
        ),
        vec3f(),
        depth_up >= 1 || depth_down >= 1,
    );
    
    var normal = cross(dPdx, dPdy);
    let normal_len = length(normal);


    normal = select(
        normal / normal_len,
        normalize(uniforms.cameraPos - pos_center), // degenerate case, use view direction
        normal_len < 1e-6,
    );
    
    // ensure normal points towards camera
    normal *= sign(dot(normal, normalize(uniforms.cameraPos - pos_center)));
    
    // store normal and compression value (J < 1 = packed, J ≈ 1 = loose)
    textureStore(normalTexture, coords, vec4f(normal, compression));
}
