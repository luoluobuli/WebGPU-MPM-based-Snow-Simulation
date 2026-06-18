// Keep this shader self-contained. The full MPM prelude contains unused helper
// functions that reference bindings this module does not declare.
struct ParticleData {
    pos: vec3f,
    _hom: f32,
    vel: vec3f,
    mass: f32,
    deformationElastic: mat3x3f,
    deformationPlastic: mat3x3f,
    pos_displacement: vec3f,
    deformation_displacement: mat3x3f,
}

const IDENTITY_MAT3 = mat3x3f(
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
);

const DEFAULT_PARTICLE_MASS = 1.0 / 3.0;
const PARTICLE_FLAG_CACHED_IDENTITY_DETERMINANTS = 4u;
const PARTICLE_MATERIAL_SHIFT = 8u;
const PARTICLE_MATERIAL_MASK = 3u << PARTICLE_MATERIAL_SHIFT;
const PARTICLE_MATERIAL_SOIL = 1u;
const PARTICLE_MATERIAL_BARK = 2u;
const PARTICLE_MATERIAL_LEAF = 3u;

fn materialFlagFromId(material: u32) -> u32 {
    return (material & 3u) << PARTICLE_MATERIAL_SHIFT;
}

fn particleMaterialFromFlags(flags: u32) -> u32 {
    return (flags & PARTICLE_MATERIAL_MASK) >> PARTICLE_MATERIAL_SHIFT;
}

fn particleMassForMaterial(material: u32) -> f32 {
    if material == PARTICLE_MATERIAL_SOIL {
        return 0.82;
    }
    if material == PARTICLE_MATERIAL_BARK {
        return 0.48;
    }
    if material == PARTICLE_MATERIAL_LEAF {
        return 0.18;
    }

    return DEFAULT_PARTICLE_MASS;
}

struct SimulationPlaybackFrameParticle {
    pos: vec3f,
    material: u32,
}

@group(0) @binding(0) var<storage, read> source_particle_data: array<ParticleData>;
@group(0) @binding(1) var<storage, read> source_particle_flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> cached_particles: array<SimulationPlaybackFrameParticle>;
@group(0) @binding(3) var<storage, read_write> restored_particle_data: array<ParticleData>;
@group(0) @binding(4) var<storage, read_write> restored_particle_flags: array<u32>;

@compute
@workgroup_size(256)
fn packSimulationPlaybackFrame(
    @builtin(global_invocation_id) global_id: vec3u,
) {
    let particle_index = global_id.x;
    if particle_index >= arrayLength(&source_particle_data) { return; }

    cached_particles[particle_index].pos = source_particle_data[particle_index].pos;
    cached_particles[particle_index].material = particleMaterialFromFlags(
        source_particle_flags[particle_index],
    );
}

@compute
@workgroup_size(256)
fn restoreSimulationPlaybackFrame(
    @builtin(global_invocation_id) global_id: vec3u,
) {
    let particle_index = global_id.x;
    if particle_index >= arrayLength(&restored_particle_data) { return; }

    let cached_particle = cached_particles[particle_index];
    let material = cached_particle.material;
    let particle = &restored_particle_data[particle_index];

    (*particle).pos = cached_particle.pos;
    (*particle)._hom = 1.0;
    (*particle).vel = vec3f();
    (*particle).mass = particleMassForMaterial(material);
    (*particle).deformationElastic = IDENTITY_MAT3;
    (*particle).deformationPlastic = IDENTITY_MAT3;
    (*particle).pos_displacement = vec3f();
    (*particle).deformation_displacement = mat3x3f();

    restored_particle_flags[particle_index] =
        materialFlagFromId(material)
        | PARTICLE_FLAG_CACHED_IDENTITY_DETERMINANTS;
}
