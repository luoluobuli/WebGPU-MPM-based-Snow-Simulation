

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid: SparseGridStorage;
// @group(1) @binding(10) is declared in colliderPrelude.wgsl (colliderSdfData)


@group(2) @binding(0) var<storage, read_write> particle_data: array<ParticleData>;
@group(2) @binding(1) var<storage, read_write> max_particle_speed_bits: atomic<u32>;

override N_PARTICLES: u32 = 0u;
override RECORD_PARTICLE_SPEED: u32 = 0u;

var<workgroup> workgroup_max_particle_speed_bits: array<u32, 256>;

fn recordWorkgroupMaxParticleSpeed(local_index: u32, speed_squared: f32) {
    workgroup_max_particle_speed_bits[local_index] = select(
        0u,
        bitcast<u32>(speed_squared),
        speed_squared > 0.0,
    );

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        workgroupBarrier();
        if local_index < stride {
            workgroup_max_particle_speed_bits[local_index] = max(
                workgroup_max_particle_speed_bits[local_index],
                workgroup_max_particle_speed_bits[local_index + stride],
            );
        }
    }

    if local_index == 0u {
        let max_speed_bits = workgroup_max_particle_speed_bits[0];
        if max_speed_bits != 0u {
            atomicMax(&max_particle_speed_bits, max_speed_bits);
        }
    }
}

fn applySimulationDomainBoundary(particle: ptr<function, ParticleData>) {
    let domain_min = uniforms.gridMinCoords;
    let domain_max = uniforms.gridMaxCoords;
    if all((*particle).pos >= domain_min) && all((*particle).pos < domain_max) {
        return;
    }

    let domain_max_inside = simulationDomainMaxInside();
    if (*particle).pos.x < domain_min.x {
        (*particle).pos_displacement.x *= -0.5;
        (*particle).pos.x = domain_min.x;
        (*particle).vel.x *= -0.5;
    }
    if (*particle).pos.x >= domain_max.x {
        (*particle).pos_displacement.x *= -0.5;
        (*particle).pos.x = domain_max_inside.x;
        (*particle).vel.x *= -0.5;
    }

    if (*particle).pos.y < domain_min.y {
        (*particle).pos_displacement.y *= -0.5;
        (*particle).pos.y = domain_min.y;
        (*particle).vel.y *= -0.5;
    }
    if (*particle).pos.y >= domain_max.y {
        (*particle).pos_displacement.y *= -0.5;
        (*particle).pos.y = domain_max_inside.y;
        (*particle).vel.y *= -0.5;
    }

    if (*particle).pos.z < domain_min.z {
        (*particle).pos_displacement.z *= -0.5;
        (*particle).pos.z = domain_min.z;
        (*particle).vel.z *= -0.5;
    }
    if (*particle).pos.z >= domain_max.z {
        (*particle).pos_displacement.z *= -0.5;
        (*particle).pos.z = domain_max_inside.z;
        (*particle).vel.z *= -0.5;
    }
}

@compute
@workgroup_size(256)
fn integrateParticles(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let particle_index = gid.x;
    let local_particle_index = lid.x;
    var speed_squared = 0.0;

    if particle_index < N_PARTICLES {
        var particle = particle_data[particle_index];
        let write_hom = !isFiniteScalar(particle._hom) || particle._hom != 1.0;
        let write_mass = !isFiniteScalar(particle.mass) || particle.mass <= 0.0;

        // let gravitational_acceleration = vec3f(0, 0, -9.81);
        // particle.pos_displacement += gravitational_acceleration * uniforms.simulationTimestep * uniforms.simulationTimestep;

        particle.pos_displacement = particle.vel * uniforms.simulationTimestep;
        particle.pos += particle.pos_displacement;

        let deformation_matrices_changed = !mat3x3IsZero(particle.deformation_displacement);
        var deformation_plastic_changed = false;
        if deformation_matrices_changed {
            particle.deformationElastic = (IDENTITY_MAT3 + particle.deformation_displacement) * particle.deformationElastic;

            let deformation_det = determinant(particle.deformationElastic);
            if deformation_det != deformation_det || deformation_det < 0.05 || deformation_det > 20.0 {
                particle.deformationElastic = IDENTITY_MAT3;
                particle.deformationPlastic = IDENTITY_MAT3;
                particle.deformation_displacement = mat3x3f();
                deformation_plastic_changed = true;
            } else {
                deformation_plastic_changed = applyPlasticity(&particle);
            }
        }

        applySimulationDomainBoundary(&particle);

        // SDF collision
        resolveParticleCollision(&particle);
        sanitizeParticleKinematicsWithoutDeformationDeltaWithKnownScalarRepairs(
            &particle,
            write_hom,
            write_mass,
        );
        if deformation_matrices_changed {
            particle.deformation_displacement = sanitizeNonZeroDeformationDelta(particle.deformation_displacement);
            let matrix_sanitize_flags = sanitizeParticleMatricesAndGetChangedFlags(&particle);
            deformation_plastic_changed = deformation_plastic_changed
                || (matrix_sanitize_flags & PARTICLE_MATRIX_PLASTIC_CHANGED) != 0u;
        }
        if RECORD_PARTICLE_SPEED != 0u {
            speed_squared = dot(particle.vel, particle.vel);
        }

        mapParticleAffectedBlocksInGrid(particle.pos, sparse_grid.current_generation);

        particle_data[particle_index].pos = particle.pos;
        particle_data[particle_index].vel = particle.vel;
        particle_data[particle_index].pos_displacement = particle.pos_displacement;
        if write_hom {
            particle_data[particle_index]._hom = particle._hom;
        }
        if write_mass {
            particle_data[particle_index].mass = particle.mass;
        }
        if deformation_matrices_changed {
            particle_data[particle_index].deformation_displacement = particle.deformation_displacement;
            particle_data[particle_index].deformationElastic = particle.deformationElastic;
            if deformation_plastic_changed {
                particle_data[particle_index].deformationPlastic = particle.deformationPlastic;
            }
        }
    }

    if RECORD_PARTICLE_SPEED != 0u {
        recordWorkgroupMaxParticleSpeed(local_particle_index, speed_squared);
    }
}
