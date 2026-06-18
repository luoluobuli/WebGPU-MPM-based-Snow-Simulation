@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> particle_data: array<ParticleData>;
@group(0) @binding(2) var<storage, read> particle_appearance: array<u32>;
@group(0) @binding(3) var<storage, read> particle_flags: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) center_world: vec3f,
    @location(2) radius: f32,
    @location(3) compression_volume_fac: f32,
    @location(4) elastic_volume_fac: f32,
    @location(5) shape_params: vec4f,
    @location(6) grain_params: vec4f,
    @location(7) appearance: vec4f,
}

const BASE_PARTICLE_RADIUS = 0.1;
const DEFAULT_SPLAT_RADIUS_SCALE = 0.88;
const SOIL_SPLAT_RADIUS_SCALE = 0.76;
const BARK_SPLAT_RADIUS_SCALE = 0.72;
const LEAF_SPLAT_RADIUS_SCALE = 1.55;
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

fn randomFromIndex(index: u32, salt: u32) -> f32 {
    return f32(hash1(index ^ salt)) / f32(0xffffffffu);
}

fn splatRadiusScaleForMaterial(material: f32) -> f32 {
    if material > 2.5 {
        return LEAF_SPLAT_RADIUS_SCALE;
    }
    if material > 1.5 {
        return BARK_SPLAT_RADIUS_SCALE;
    }
    if material > 0.5 {
        return SOIL_SPLAT_RADIUS_SCALE;
    }

    return DEFAULT_SPLAT_RADIUS_SCALE;
}

@vertex
fn vert(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let particle = particle_data[instance_index];
    let flags = particle_flags[instance_index];
    let appearance = decodeParticleAppearance(particle_appearance[instance_index]);
    let uv = VERT_POSITIONS[vertex_index];
    let seed = vec3f(
        randomFromIndex(instance_index, 0x9e3779b9u),
        randomFromIndex(instance_index, 0x85ebca6bu),
        randomFromIndex(instance_index, 0xc2b2ae35u),
    );
    let radius_jitter = mix(0.72, 1.08, seed.x);
    let radius = BASE_PARTICLE_RADIUS
        * particle.mass
        * splatRadiusScaleForMaterial(appearance.w)
        * radius_jitter;
    let axis_scale = vec2f(
        mix(0.72, 1.18, seed.y),
        mix(0.68, 1.12, seed.z),
    );
    let depth_scale = mix(0.48, 0.72, randomFromIndex(instance_index, 0x27d4eb2fu));
    let grain_params = vec4f(
        randomFromIndex(instance_index, 0x165667b1u),
        randomFromIndex(instance_index, 0xd3a2646cu),
        randomFromIndex(instance_index, 0xfd7046c5u),
        randomFromIndex(instance_index, 0xb55a4f09u),
    );
    let basis = cameraBillboardBasis(particle.pos);
    let vertex_pos_world = particle.pos + (basis[0] * uv.x * axis_scale.x + basis[1] * uv.y * axis_scale.y) * radius;

    var out: VertexOutput;
    out.position = uniforms.viewProjMat * vec4f(vertex_pos_world, 1);
    out.uv = uv;
    out.center_world = particle.pos;
    out.radius = radius;
    out.compression_volume_fac = particlePlasticVolumeFactor(particle, flags);
    out.elastic_volume_fac = particleElasticVolumeFactor(particle, flags);
    out.shape_params = vec4f(axis_scale, depth_scale, seed.x);
    out.grain_params = grain_params;
    out.appearance = appearance;

    return out;
}
