struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) center_view: vec3f,
    @location(2) radius: f32,
    @location(3) center_world: vec3f,
}

struct FragOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4f,
}

@fragment
fn frag(in: VertexOutput) -> FragOutput {
    let r2 = dot(in.uv, in.uv);
    if (r2 > 1) {
        discard;
    }

    let camera_pos = uniforms.cameraPos;
    
    let forward = normalize(camera_pos - in.center_world);
    let right = normalize(cross(vec3f(0, 0, 1), forward));
    let up = normalize(cross(forward, right));
    
    let z = sqrt(1 - r2);
    let normal_world = right * in.uv.x + up * in.uv.y + forward * z;
    let surface_pos_world = in.center_world + normal_world * in.radius;
    
    let clip_pos = uniforms.viewProjMat * vec4f(surface_pos_world, 1);
    let depth = clip_pos.z / clip_pos.w;
    
    var out: FragOutput;

    out.depth = depth;
    
    let z_near = 0.01;
    let z_far = 100.;
    let linear_depth = (2 * z_near * z_far) / (z_far + z_near - (2 * depth - 1) * (z_far - z_near));
    
    out.color = vec4f(vec3f(linear_depth / 10), 1);
    
    return out;
}
