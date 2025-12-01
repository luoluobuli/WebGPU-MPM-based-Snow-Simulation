@group(0) @binding(1) var inputTexture: texture_2d<f32>;
@group(0) @binding(2) var depthTexture: texture_2d<f32>;

struct FragmentOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn frag(in: VertexOut) -> FragmentOutput {
    let dims = textureDimensions(inputTexture);
    let coords = vec2u(in.uv * vec2f(dims));
    
    var output: FragmentOutput;

    output.color = textureLoad(inputTexture, coords, 0);
    let world_depth = textureLoad(depthTexture, coords, 0).r;

    if world_depth > 1e12 {
        output.depth = 1;
    } else {
        let uvNormalized = in.uv * 2 - 1;
        let uvFlipped = vec2f(uvNormalized.x, -uvNormalized.y);
        
        let nearPosHom = uniforms.viewProjInvMat * vec4f(uvFlipped, 0, 1);
        let nearPos = nearPosHom.xyz / nearPosHom.w;
        
        let farPosHom = uniforms.viewProjInvMat * vec4f(uvFlipped, 1, 1);
        let farPos = farPosHom.xyz / farPosHom.w;
        
        let ray_origin = nearPos;
        let ray_dir = normalize(farPos - nearPos);
        
        let world_pos = ray_origin + ray_dir * world_depth;
        
        let clip_pos = uniforms.viewProjMat * vec4f(world_pos, 1);
        
        output.depth = clip_pos.z / clip_pos.w;
    }
    
    return output;
}
