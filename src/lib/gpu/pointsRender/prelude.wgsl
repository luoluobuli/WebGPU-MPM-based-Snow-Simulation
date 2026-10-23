struct PointsVertexOut {
    @builtin(position) posBuiltin: vec4f,
    @location(0) pos: vec4f,
    @location(1) uv: vec2f,
    @location(2) deformation_elastic_volume: f32,
    @location(3) deformation_plastic_volume: f32,
}
