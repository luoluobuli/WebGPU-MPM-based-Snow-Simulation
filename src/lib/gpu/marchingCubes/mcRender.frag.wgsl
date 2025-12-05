// Marching cubes mesh fragment shader - G-buffer output

struct FragmentInput {
    @location(0) worldPos: vec3f,
    @location(1) normal: vec3f,
}

struct FragmentOutput {
    @location(0) albedo: vec4f,
    @location(1) normalDepth: vec4f,
}

@fragment
fn frag(input: FragmentInput) -> FragmentOutput {
    var output: FragmentOutput;
    
    // Snow albedo - slightly bluish white
    output.albedo = vec4f(0.95, 0.97, 1.0, 1.0);
    
    // Pack normal and write
    let normalNormalized = normalize(input.normal);
    output.normalDepth = vec4f(normalNormalized, 1.0);
    
    return output;
}
