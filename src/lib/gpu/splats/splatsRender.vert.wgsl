@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> particle_data: array<ParticleData>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) center_world: vec3f,
    @location(2) radius: f32,
    @location(3) compression_volume_fac: f32,
    @location(4) elastic_volume_fac: f32,
}

const BASE_PARTICLE_RADIUS = 0.1;
const SPLAT_RADIUS_SCALE = 1.35;
const VERT_POSITIONS: array<vec2f, 6> = array(
    vec2f(-1, -1),
    vec2f(1, -1),
    vec2f(1, 1),
    vec2f(-1, -1),
    vec2f(1, 1),
    vec2f(-1, 1),
);

fn cameraBillboardBasis(center_world: vec3f) -> mat3x3f {
    let forward = normalize(uniforms.cameraPos - center_world);
    let reference_up = select(vec3f(0, 0, 1), vec3f(0, 1, 0), abs(forward.z) > 0.95);
    let right = normalize(cross(reference_up, forward));
    let up = cross(forward, right);

    return mat3x3f(right, up, forward);
}

@vertex
fn vert(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let particle = particle_data[instance_index];
    let uv = VERT_POSITIONS[vertex_index];
    let radius = BASE_PARTICLE_RADIUS * particle.mass * SPLAT_RADIUS_SCALE;
    let basis = cameraBillboardBasis(particle.pos);
    let vertex_pos_world = particle.pos + (basis[0] * uv.x + basis[1] * uv.y) * radius;

    var out: VertexOutput;
    out.position = uniforms.viewProjMat * vec4f(vertex_pos_world, 1);
    out.uv = uv;
    out.center_world = particle.pos;
    out.radius = radius;
    out.compression_volume_fac = determinant(particle.deformationPlastic);
    out.elastic_volume_fac = determinant(particle.deformationElastic);

    return out;
}
