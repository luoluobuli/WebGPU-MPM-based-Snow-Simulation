@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read> particle_data: array<ParticleData>;
@group(1) @binding(1) var<storage, read_write> max_particle_speed_bits: atomic<u32>;

@compute
@workgroup_size(256)
fn reduceMaxParticleSpeed(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let particle_index = gid.x;
    if particle_index >= arrayLength(&particle_data) { return; }

    let particle = particle_data[particle_index];
    let safe_velocity = sanitizeVec3(particle.vel, vec3f(0.0));
    let speed_squared = dot(safe_velocity, safe_velocity);
    if isFiniteScalar(speed_squared) && speed_squared > 0.0 {
        atomicMax(&max_particle_speed_bits, bitcast<u32>(speed_squared));
    }
}
