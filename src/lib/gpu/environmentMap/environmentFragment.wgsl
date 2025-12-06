@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var environment_texture: texture_2d<f32>;

fn normalizedSphericalFromRayDir(ray_dir: vec3f) -> vec2f {
    let theta = (atan2(ray_dir.y, ray_dir.x) + PI) / (2 * PI);
    let phi = acos(ray_dir.z) / PI;
    return vec2f(theta, phi);
}


@fragment
fn frag(in: VertexOut) -> @location(0) vec4f {
    let dims = textureDimensions(environment_texture);

    let ray = calculateViewRay(in.uv, vec2u(dims));
    let spherical_uv = normalizedSphericalFromRayDir(ray.dir);

    return textureLoad(environment_texture, vec2u(spherical_uv * vec2f(dims)), 0);
}
