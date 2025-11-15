@fragment
fn frag(
    in: RaymarchVertexOut,
) -> @location(0) vec4f {
    let rayDir = uniforms.viewInvMat * vec4f(normalize(vec3f(in.uvCentered, -1)), 0);

    return vec4f(rayDir.xyz * 0.5 + 0.5, 1);
}