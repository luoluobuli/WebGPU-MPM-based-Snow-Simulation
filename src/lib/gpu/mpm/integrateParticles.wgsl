

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid: SparseGridStorage;
// @group(1) @binding(10) is declared in colliderPrelude.wgsl (colliderSdfData)


@group(2) @binding(0) var<storage, read_write> particle_data: array<ParticleData>;
@group(2) @binding(1) var<storage, read_write> max_particle_speed_bits: atomic<u32>;
@group(2) @binding(2) var<storage, read_write> particle_flags: array<u32>;

override N_PARTICLES: u32 = 0u;
override RECORD_PARTICLE_SPEED: u32 = 0u;
override PARTICLE_WORKGROUP_SIZE: u32 = 256u;
override PARTICLE_SPEED_REDUCTION_WORKGROUP_SIZE: u32 = 1u;

var<workgroup> workgroup_max_particle_speed_bits: array<u32, PARTICLE_SPEED_REDUCTION_WORKGROUP_SIZE>;

fn recordWorkgroupMaxParticleSpeed(local_index: u32, speed_squared: f32) {
    workgroup_max_particle_speed_bits[local_index] = select(
        0u,
        bitcast<u32>(speed_squared),
        speed_squared > 0.0,
    );

    for (var stride = PARTICLE_SPEED_REDUCTION_WORKGROUP_SIZE >> 1u; stride > 0u; stride = stride >> 1u) {
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

fn applySimulationDomainBoundary(
    particle: ptr<function, ParticleData>,
    material: u32,
) {
    let domain_min = uniforms.gridMinCoords;
    let domain_max = uniforms.gridMaxCoords;
    if all((*particle).pos >= domain_min) && all((*particle).pos < domain_max) {
        return;
    }

    let domain_max_inside = simulationDomainMaxInside();
    let tangential_scale = 1.0 - materialBoundaryFriction(material);
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
        (*particle).vel.x *= tangential_scale;
        (*particle).vel.y *= tangential_scale;
    }
    if (*particle).pos.z >= domain_max.z {
        (*particle).pos_displacement.z *= -0.5;
        (*particle).pos.z = domain_max_inside.z;
        (*particle).vel.z *= -0.5;
    }
}

@compute
@workgroup_size(PARTICLE_WORKGROUP_SIZE)
fn integrateParticles(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let particle_index = gid.x;
    let local_particle_index = lid.x;
    var speed_squared = 0.0;

    if particle_index < N_PARTICLES {
        let sparse_grid_generation = sparse_grid.current_generation;
        var particle = particle_data[particle_index];
        var flags = particle_flags[particle_index];
        let material = particleMaterial(flags);

        if particleIsAnchored(flags) {
            resetAnchoredParticleMotion(&particle);
            mapParticleAffectedBlocksInGrid(particle.pos, sparse_grid_generation);

            particle_data[particle_index].vel = particle.vel;
            particle_data[particle_index].pos_displacement = particle.pos_displacement;
            particle_data[particle_index].deformation_displacement = particle.deformation_displacement;
            particle_data[particle_index].deformationElastic = particle.deformationElastic;
            particle_data[particle_index].deformationPlastic = particle.deformationPlastic;
            particle_flags[particle_index] = particlePersistentFlags(flags) & ~PARTICLE_FLAG_ELASTIC_NON_IDENTITY;
        } else {
            applyMaterialVelocityDamping(&particle, material);
            let unclamped_velocity = sanitizeVec3(particle.vel, vec3f(0.0));
            let unclamped_speed_squared = dot(unclamped_velocity, unclamped_velocity);
            particle.vel = clampVec3LengthNoSanitizeWithMaxSquared(
                unclamped_velocity,
                uniforms.maxStableParticleSpeed,
                uniforms.maxStableParticleSpeedSquared,
            );

            // let gravitational_acceleration = vec3f(0, 0, -9.81);
            // particle.pos_displacement += gravitational_acceleration * uniforms.simulationTimestep * uniforms.simulationTimestep;

            particle.pos_displacement = particle.vel * uniforms.simulationTimestep;
            particle.pos += particle.pos_displacement;

            let deformation_matrices_changed = (flags & PARTICLE_FLAG_DEFORMATION_DELTA_VALID) != 0u;
            var deformation_delta_remains_valid = deformation_matrices_changed;
            var deformation_plastic_changed = false;
            if deformation_matrices_changed {
                let deformation_delta = deformationDeltaFromVelocityGradient(particle.deformation_displacement);
                if mat3x3IsZero(deformation_delta) {
                    particle.deformation_displacement = mat3x3f();
                    deformation_delta_remains_valid = false;
                } else {
                    particle.deformationElastic = (IDENTITY_MAT3 + deformation_delta) * particle.deformationElastic;
                }

                let deformation_det = determinant(particle.deformationElastic);
                if deformation_det != deformation_det || deformation_det < 0.05 || deformation_det > 20.0 {
                    particle.deformationElastic = IDENTITY_MAT3;
                    particle.deformationPlastic = IDENTITY_MAT3;
                    particle.deformation_displacement = mat3x3f();
                    deformation_delta_remains_valid = false;
                    deformation_plastic_changed = true;
                } else {
                    deformation_plastic_changed = applyPlasticity(&particle, material);
                }
            }

            applySimulationDomainBoundary(&particle, material);

            // SDF collision
            resolveParticleCollision(&particle);
            sanitizeParticleKinematicsWithoutDeformationDeltaWithKnownScalarRepairs(
                &particle,
                false,
                false,
            );
            if deformation_matrices_changed {
                particle.deformation_displacement = sanitizeVelocityGradient(particle.deformation_displacement);
                let matrix_sanitize_flags = sanitizeParticleMatricesAndGetChangedFlags(&particle);
                deformation_plastic_changed = deformation_plastic_changed
                    || (matrix_sanitize_flags & PARTICLE_MATRIX_PLASTIC_CHANGED) != 0u;
                deformation_delta_remains_valid = !mat3x3IsZero(particle.deformation_displacement);
            }
            if RECORD_PARTICLE_SPEED != 0u {
                speed_squared = unclamped_speed_squared;
            }

            mapParticleAffectedBlocksInGrid(particle.pos, sparse_grid_generation);

            particle_data[particle_index].pos = particle.pos;
            particle_data[particle_index].vel = particle.vel;
            if deformation_matrices_changed {
                particle_data[particle_index].deformation_displacement = particle.deformation_displacement;
                particle_data[particle_index].deformationElastic = particle.deformationElastic;
                if deformation_plastic_changed {
                    particle_data[particle_index].deformationPlastic = particle.deformationPlastic;
                }
                if mat3x3IsIdentity(particle.deformationElastic) {
                    flags = flags & ~PARTICLE_FLAG_ELASTIC_NON_IDENTITY;
                } else {
                    flags = flags | PARTICLE_FLAG_ELASTIC_NON_IDENTITY;
                }
                if !deformation_delta_remains_valid {
                    flags = flags & ~PARTICLE_FLAG_DEFORMATION_DELTA_VALID;
                }
                particle_flags[particle_index] = flags;
            }
        }
    }

    if RECORD_PARTICLE_SPEED != 0u {
        recordWorkgroupMaxParticleSpeed(local_particle_index, speed_squared);
    }
}
