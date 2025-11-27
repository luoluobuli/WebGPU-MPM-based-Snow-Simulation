struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

fn linearSplineWeights(fractional_pos: vec3f) -> array<vec3f, 2> {
    return array(vec3f(1 - fractional_pos), vec3f(fractional_pos));
}

const MASS_FIXED_POINT_SCALE = 1000.;
