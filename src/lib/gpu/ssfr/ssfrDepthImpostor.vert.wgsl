@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> particle_data: array<ParticleData>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) center_view: vec3f,
    @location(2) sphere_radius: f32,
    @location(3) center_world: vec3f,
    @location(4) quad_scale: f32,
    @location(5) compression_volume_fac: f32,
}

const BASE_PARTICLE_RADIUS = 0.1;
const OVERLAP_FACTOR = 2.;
const VERT_POSITIONS: array<vec2f, 6> = array(
    vec2f(-1, -1),
    vec2f(1, -1),
    vec2f(1, 1),
    vec2f(-1, -1),
    vec2f(1, 1),
    vec2f(-1, 1),
);

@vertex
fn vert(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    var out: VertexOutput;
    let particle = particle_data[instance_index];
    let pos_world = vec4f(particle.pos, 1);

    let radius = BASE_PARTICLE_RADIUS * particle.mass;
    
    let sphere_radius = radius;
    let quad_radius = radius * OVERLAP_FACTOR;
    out.sphere_radius = sphere_radius;
    out.quad_scale = OVERLAP_FACTOR;
    out.center_world = particle.pos;

    let camera_pos = uniforms.cameraPos;
    let forward = normalize(camera_pos - particle.pos);
    let right = normalize(cross(vec3f(0, 0, 1), forward));
    let up = normalize(cross(forward, right));
    
    let uv = VERT_POSITIONS[vertex_index];
    
    out.uv = uv;
    
    let vertex_pos_world = particle.pos + (right * uv.x + up * uv.y) * quad_radius;
    out.position = uniforms.viewProjMat * vec4f(vertex_pos_world, 1);
    
    out.compression_volume_fac = determinant(particle.deformationPlastic);
    
    return out;
}
