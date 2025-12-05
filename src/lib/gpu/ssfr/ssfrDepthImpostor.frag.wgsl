struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) center_view: vec3f,
    @location(2) sphere_radius: f32,
    @location(3) center_world: vec3f,
    @location(4) quad_scale: f32,
    @location(5) compression_volume_fac: f32,
}

struct FragmentOutput {
    @location(0) mask: vec4f,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn frag(in: VertexOutput) -> FragmentOutput {
    let distance_squared = dot(in.uv, in.uv);
    if distance_squared > 1 / (in.quad_scale * in.quad_scale) { discard; }

    let sphere_uv = in.uv * in.quad_scale;
    
    let forward = normalize(uniforms.cameraPos - in.center_world);
    let right = normalize(cross(vec3f(0, 0, 1), forward));
    let up = normalize(cross(forward, right));
    
    let radius_squared = dot(sphere_uv, sphere_uv);
    let z = sqrt(max(0, 1 - radius_squared));
    let normal_world = right * sphere_uv.x + up * sphere_uv.y + forward * z;
    let surface_pos_world = in.center_world + normal_world * in.sphere_radius;
    
    let clip_pos = uniforms.viewProjMat * vec4f(surface_pos_world, 1);
    let depth = clip_pos.z / clip_pos.w;
    
    var out: FragmentOutput;
    out.mask = vec4f(depth, in.compression_volume_fac, 0, 1);
    let depth_squared = depth * depth;
    out.depth = depth_squared * depth_squared;
    return out;
}

