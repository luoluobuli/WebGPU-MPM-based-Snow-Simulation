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

const DEFAULT_PARTICLE_MASS = 1.0 / 3.0;
const PARTICLE_FLAG_CACHED_IDENTITY_DETERMINANTS = 4u;

@group(0) @binding(0) var<storage, read> source_particle_data: array<ParticleData>;
@group(0) @binding(2) var<storage, read_write> cached_particles: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> restored_particle_data: array<ParticleData>;
@group(0) @binding(4) var<storage, read_write> restored_particle_flags: array<u32>;

@compute
@workgroup_size(256)
fn packSimulationPlaybackFrame(
    @builtin(global_invocation_id) global_id: vec3u,
) {
    let particle_index = global_id.x;
    if particle_index >= arrayLength(&source_particle_data) { return; }

    cached_particles[particle_index] = vec4f(
        source_particle_data[particle_index].pos,
        1.0,
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
    let particle = &restored_particle_data[particle_index];

    // Cached playback frames are display snapshots. Solver state is rebuilt from
    // scatter before baking again, so restore only fields the renderers consume.
    (*particle).pos = cached_particle.xyz;
    (*particle)._hom = 1.0;
    (*particle).mass = DEFAULT_PARTICLE_MASS;

    restored_particle_flags[particle_index] = PARTICLE_FLAG_CACHED_IDENTITY_DETERMINANTS;
}
