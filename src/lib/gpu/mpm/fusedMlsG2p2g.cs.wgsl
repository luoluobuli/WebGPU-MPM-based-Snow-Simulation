@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> sparse_grid: SparseGridStorage;

struct GridAccumulatorAtomic {
    mass: atomic<i32>,
    momentum_x: atomic<i32>,
    momentum_y: atomic<i32>,
    momentum_z: atomic<i32>,
}

@group(1) @binding(3) var<storage, read_write> grid_accumulator: array<GridAccumulatorAtomic>;

@group(2) @binding(0) var<storage, read_write> particle_data: array<ParticleData>;
@group(2) @binding(1) var<storage, read_write> max_particle_speed_bits: atomic<u32>;
@group(2) @binding(2) var<storage, read_write> particle_flags: array<u32>;

struct BukkitThreadData {
    range_start: u32,
    range_count: u32,
    origin_cell_x: u32,
    origin_cell_y: u32,
    origin_cell_z: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct BukkitDispatchArgsRead {
    x: u32,
    y: u32,
    z: u32,
    count: u32,
}

struct BukkitThreadGroupCountRead {
    count: u32,
}

@group(3) @binding(0) var<storage, read_write> next_sparse_grid: SparseGridStorage;

@group(3) @binding(3) var<storage, read_write> next_grid_accumulator: array<GridAccumulatorAtomic>;
@group(3) @binding(5) var<storage, read_write> bukkit_thread_data: array<BukkitThreadData>;
@group(3) @binding(6) var<storage, read_write> bukkit_particle_data: array<u32>;
@group(3) @binding(9) var<storage, read_write> bukkit_thread_group_count: BukkitThreadGroupCountRead;

override N_PARTICLES: u32 = 0u;
override PARTICLE_WORKGROUP_SIZE: u32 = 256u;
override ENABLE_INTERACTION: u32 = 1u;
override GRID_BOUNDARY_MAX_X: i32 = 381i;
override GRID_BOUNDARY_MAX_Y: i32 = 381i;
override GRID_BOUNDARY_MAX_Z: i32 = 381i;
override GRID_BOUNDARY_HIGH_BLOCK_X: i32 = 95i;
override GRID_BOUNDARY_HIGH_BLOCK_Y: i32 = 95i;
override GRID_BOUNDARY_HIGH_BLOCK_Z: i32 = 95i;
override RECORD_PARTICLE_SPEED: u32 = 0u;
override PARTICLE_SPEED_REDUCTION_WORKGROUP_SIZE: u32 = 256u;

const GRID_BOUNDARY_WIDTH = 2i;
const FUSED_BUKKIT_SIZE = 2u;
const FUSED_BUKKIT_HALO = 1u;
const FUSED_TILE_EDGE = FUSED_BUKKIT_SIZE + 2u * FUSED_BUKKIT_HALO;
const FUSED_TILE_CELL_COUNT = FUSED_TILE_EDGE * FUSED_TILE_EDGE * FUSED_TILE_EDGE;
const FUSED_TILE_ACCUMULATOR_SIZE = FUSED_TILE_CELL_COUNT * 4u;
const BUKKIT_DISPATCH_WIDTH = 256u;

var<workgroup> tile_velocity: array<vec4f, FUSED_TILE_CELL_COUNT>;
var<workgroup> tile_accumulator: array<atomic<i32>, FUSED_TILE_ACCUMULATOR_SIZE>;
var<workgroup> workgroup_local_grid_origin: vec3i;
var<workgroup> workgroup_has_thread_data: bool;
var<workgroup> workgroup_max_particle_speed_bits: array<u32, PARTICLE_SPEED_REDUCTION_WORKGROUP_SIZE>;

struct FusedGatherResult {
    velocity: vec3f,
    deformation_displacement: mat3x3f,
    has_velocity_contribution: bool,
}

fn tileOffsetFromIndex(index: u32) -> vec3i {
    return vec3i(
        i32(index % FUSED_TILE_EDGE),
        i32((index / FUSED_TILE_EDGE) % FUSED_TILE_EDGE),
        i32(index / (FUSED_TILE_EDGE * FUSED_TILE_EDGE)),
    );
}

fn tileIndexFromOffset(offset: vec3i) -> u32 {
    let o = vec3u(offset);
    return o.x + FUSED_TILE_EDGE * (o.y + FUSED_TILE_EDGE * o.z);
}

fn tileOffsetIsInRange(offset: vec3i) -> bool {
    return all(offset >= vec3i(0)) && all(offset < vec3i(i32(FUSED_TILE_EDGE)));
}

fn tileAccumulatorIndex(tile_index: u32) -> u32 {
    return tile_index * 4u;
}

fn cellOffsetWithinBlock(cell_number: vec3i) -> vec3u {
    return vec3u(cell_number & vec3i(i32(BLOCK_MASK)));
}

fn cellCenterGridCoord(cell_number: vec3i) -> vec3f {
    return vec3f(cell_number) + vec3f(0.5);
}

fn boundaryLowEnd(block_axis: i32) -> u32 {
    return u32(clamp(
        GRID_BOUNDARY_WIDTH - block_axis * i32(BLOCK_SIZE),
        0i,
        i32(BLOCK_SIZE),
    ));
}

fn boundaryHighStart(block_axis: i32, boundary_max: i32) -> u32 {
    return u32(clamp(
        boundary_max - block_axis * i32(BLOCK_SIZE) + 1i,
        0i,
        i32(BLOCK_SIZE),
    ));
}

fn applyInteractionVelocity(
    velocity: vec3f,
    cell_number: vec3i,
) -> vec3f {
    let grid_pos = cellCenterGridCoord(cell_number);
    let offset = grid_pos - uniforms.interactionPos;
    let dist_squared = dot(offset, offset);
    let radius_squared = uniforms.interactionRadiusSquared;

    if dist_squared < radius_squared {
        var dir = vec3f(0.0, 0.0, 1.0);
        if dist_squared > 1e-6 {
            dir = offset * inverseSqrt(dist_squared);
        }

        let falloff = 1.0 - dist_squared / radius_squared;
        let signed_strength = select(1.0, -1.0, uniforms.interactionMode == 1u);
        return velocity + signed_strength * dir * uniforms.interactionStrengthDelta * falloff;
    }

    return velocity;
}

fn applyDomainBoundaryVelocity(
    velocity: vec3f,
    block_number: vec3i,
    cell_offset: vec3u,
) -> vec3f {
    let boundary_low_end = vec3u(
        boundaryLowEnd(block_number.x),
        boundaryLowEnd(block_number.y),
        boundaryLowEnd(block_number.z),
    );
    let boundary_high_start = vec3u(
        boundaryHighStart(block_number.x, GRID_BOUNDARY_MAX_X),
        boundaryHighStart(block_number.y, GRID_BOUNDARY_MAX_Y),
        boundaryHighStart(block_number.z, GRID_BOUNDARY_MAX_Z),
    );

    var bounded_velocity = velocity;
    if cell_offset.x < boundary_low_end.x && bounded_velocity.x < 0.0 { bounded_velocity.x = 0.0; }
    if cell_offset.x >= boundary_high_start.x && bounded_velocity.x > 0.0 { bounded_velocity.x = 0.0; }
    if cell_offset.y < boundary_low_end.y && bounded_velocity.y < 0.0 { bounded_velocity.y = 0.0; }
    if cell_offset.y >= boundary_high_start.y && bounded_velocity.y > 0.0 { bounded_velocity.y = 0.0; }
    if cell_offset.z < boundary_low_end.z && bounded_velocity.z < 0.0 { bounded_velocity.z = 0.0; }
    if cell_offset.z >= boundary_high_start.z && bounded_velocity.z > 0.0 { bounded_velocity.z = 0.0; }

    return bounded_velocity;
}

fn clampVelocityToFixedPointRange(velocity: vec3f) -> vec3f {
    return clampVec3LengthWithMaxSquared(
        velocity,
        maxFixedPointGridSpeed(),
        maxFixedPointGridSpeedSquared(),
    );
}

fn loadTileVelocity(cell_number: vec3i) -> vec3f {
    if !cellNumberInSparseGridRange(cell_number) {
        return vec3f(0.0);
    }

    let cell_index = calculateCellIndexFromCellNumber(cell_number);
    if cell_index == GRID_BLOCK_INDEX_EMPTY {
        return vec3f(0.0);
    }

    let cell_mass_fixed_i32 = atomicLoad(&grid_accumulator[cell_index].mass);
    if cell_mass_fixed_i32 <= 0i {
        return vec3f(0.0);
    }

    let inv_cell_mass_fixed = 1.0 / f32(cell_mass_fixed_i32);
    var cell_velocity = vec3f(vec3i(
        atomicLoad(&grid_accumulator[cell_index].momentum_x),
        atomicLoad(&grid_accumulator[cell_index].momentum_y),
        atomicLoad(&grid_accumulator[cell_index].momentum_z),
    )) * inv_cell_mass_fixed + uniforms.gravityDeltaVelocity;

    let block_number = calculateBlockNumberContainingCell(cell_number);
    let is_interior_block = block_number.x > 0i && block_number.x < GRID_BOUNDARY_HIGH_BLOCK_X
        && block_number.y > 0i && block_number.y < GRID_BOUNDARY_HIGH_BLOCK_Y
        && block_number.z > 0i && block_number.z < GRID_BOUNDARY_HIGH_BLOCK_Z;
    if ENABLE_INTERACTION != 0u && uniforms.isInteracting != 0u && uniforms.interactionRadiusSquared > 0.0 {
        cell_velocity = applyInteractionVelocity(cell_velocity, cell_number);
        if !is_interior_block {
            cell_velocity = applyDomainBoundaryVelocity(
                cell_velocity,
                block_number,
                cellOffsetWithinBlock(cell_number),
            );
        }
    } else if !is_interior_block {
        cell_velocity = applyDomainBoundaryVelocity(
            cell_velocity,
            block_number,
            cellOffsetWithinBlock(cell_number),
        );
    }

    return clampVelocityToFixedPointRange(cell_velocity);
}

fn velocityFromTile(cell_number: vec3i, local_grid_origin: vec3i) -> vec3f {
    let tile_offset = cell_number - local_grid_origin;
    if !tileOffsetIsInRange(tile_offset) {
        return vec3f(0.0);
    }

    return tile_velocity[tileIndexFromOffset(tile_offset)].xyz;
}

fn retrieveBlockIndexFromNextGrid(block_number: vec3i, target_generation: u32) -> u32 {
    if !bukkitCanContainBlock(block_number) {
        return GRID_BLOCK_INDEX_EMPTY;
    }

    let bukkit_index = calculateBukkitIndex(block_number);
    let bukkit_generation = atomicLoad(&next_sparse_grid.bukkit_generations[bukkit_index]);
    if bukkit_generation != target_generation {
        return GRID_BLOCK_INDEX_EMPTY;
    }

    let block_index = atomicLoad(&next_sparse_grid.block_index_bukkits[bukkit_index]);
    if block_index >= N_MAX_ACTIVE_BLOCKS {
        return GRID_BLOCK_INDEX_EMPTY;
    }

    return block_index;
}

fn allocateBlockInNextGridAndGetIndex(block_number: vec3i) -> u32 {
    if !bukkitCanContainBlock(block_number) {
        return GRID_BLOCK_INDEX_EMPTY;
    }

    let target_generation = next_sparse_grid.current_generation;
    let bukkit_index = calculateBukkitIndex(block_number);

    for (var i = 0u; i < 256u; i = i + 1u) {
        let current_generation = atomicLoad(&next_sparse_grid.bukkit_generations[bukkit_index]);

        if current_generation == target_generation {
            let block_index = atomicLoad(&next_sparse_grid.block_index_bukkits[bukkit_index]);
            if block_index < N_MAX_ACTIVE_BLOCKS {
                return block_index;
            }
            return GRID_BLOCK_INDEX_EMPTY;
        }

        if current_generation == BUKKIT_GENERATION_RESERVED {
            continue;
        }

        let res = atomicCompareExchangeWeak(
            &next_sparse_grid.bukkit_generations[bukkit_index],
            current_generation,
            BUKKIT_GENERATION_RESERVED,
        );

        if res.exchanged {
            let next_block_index = atomicAdd(&next_sparse_grid.n_allocated_blocks, 1u);

            if next_block_index >= N_MAX_ACTIVE_BLOCKS {
                atomicStore(&next_sparse_grid.block_index_bukkits[bukkit_index], GRID_BLOCK_INDEX_EMPTY);
                atomicStore(&next_sparse_grid.bukkit_generations[bukkit_index], target_generation);
                return GRID_BLOCK_INDEX_EMPTY;
            }

            next_sparse_grid.mapped_block_numbers[next_block_index] = block_number;
            atomicStore(&next_sparse_grid.block_index_bukkits[bukkit_index], next_block_index);
            atomicStore(&next_sparse_grid.bukkit_generations[bukkit_index], target_generation);
            return next_block_index;
        }
    }

    return retrieveBlockIndexFromNextGrid(block_number, target_generation);
}

fn accumulateFixedUnitsToNextGridCell(
    cell_number: vec3i,
    momentum_i: vec3i,
    mass_i: i32,
) {
    if !cellNumberInSparseGridRange(cell_number) { return; }

    let block_number = calculateBlockNumberContainingCell(cell_number);
    let block_index = allocateBlockInNextGridAndGetIndex(block_number);
    if block_index == GRID_BLOCK_INDEX_EMPTY { return; }

    let cell_index = (block_index << LOG_BLOCK_SIZE_CUBED)
        + calculateCellIndexWithinBlock(cell_number);
    atomicAdd(&next_grid_accumulator[cell_index].mass, mass_i);
    atomicAdd(&next_grid_accumulator[cell_index].momentum_x, momentum_i.x);
    atomicAdd(&next_grid_accumulator[cell_index].momentum_y, momentum_i.y);
    atomicAdd(&next_grid_accumulator[cell_index].momentum_z, momentum_i.z);
}

fn accumulateFixedUnitsToTileOrNextGrid(
    cell_number: vec3i,
    local_grid_origin: vec3i,
    momentum_fixed_units: vec3f,
    mass_fixed_units: f32,
) {
    let momentum_i = vec3i(momentum_fixed_units);
    let mass_i = i32(mass_fixed_units);
    if mass_i == 0i && all(momentum_i == vec3i(0)) {
        return;
    }

    let tile_offset = cell_number - local_grid_origin;
    if tileOffsetIsInRange(tile_offset) {
        let base_index = tileAccumulatorIndex(tileIndexFromOffset(tile_offset));
        atomicAdd(&tile_accumulator[base_index], mass_i);
        atomicAdd(&tile_accumulator[base_index + 1u], momentum_i.x);
        atomicAdd(&tile_accumulator[base_index + 2u], momentum_i.y);
        atomicAdd(&tile_accumulator[base_index + 3u], momentum_i.z);
        return;
    }

    accumulateFixedUnitsToNextGridCell(cell_number, momentum_i, mass_i);
}

fn clampMomentumFixedUnitsForAtomic(value: vec3f, max_component: f32) -> vec3f {
    return clamp(value, vec3f(-max_component), vec3f(max_component));
}

fn gatherMlsFromTile(particle_pos: vec3f, local_grid_origin: vec3i) -> FusedGatherResult {
    let particle_grid_coord = calculateGridCoordinate(particle_pos);
    if !gridCoordinateCanTouchGrid(particle_grid_coord) {
        return FusedGatherResult(vec3f(0.0), mat3x3f(), false);
    }

    let start_cell_number = vec3i(floor(particle_grid_coord));
    let cell_frac_pos = particle_grid_coord - vec3f(start_cell_number);
    let cell_weights = calculateQuadraticBSplineCellWeightVectors(cell_frac_pos);
    let stencil_weight_x = cell_weights.x;
    let stencil_weight_y = cell_weights.y;
    let stencil_weight_z = cell_weights.z;
    let stencil_offset_x = vec3f(-0.5, 0.5, 1.5) - vec3f(cell_frac_pos.x);
    let stencil_offset_y = vec3f(-0.5, 0.5, 1.5) - vec3f(cell_frac_pos.y);
    let stencil_offset_z = vec3f(-0.5, 0.5, 1.5) - vec3f(cell_frac_pos.z);

    var new_particle_velocity = vec3f(0.0);
    var b_matrix = mat3x3f();
    var has_velocity_contribution = false;

    for (var offset_z = -1i; offset_z <= 1i; offset_z = offset_z + 1i) {
        let cell_z = start_cell_number.z + offset_z;
        let z_index = u32(offset_z + 1i);
        let wz = stencil_weight_z[z_index];
        let stencil_offset_z_value = stencil_offset_z[z_index];
        for (var offset_y = -1i; offset_y <= 1i; offset_y = offset_y + 1i) {
            let cell_y = start_cell_number.y + offset_y;
            let y_index = u32(offset_y + 1i);
            let wy_wz = stencil_weight_y[y_index] * wz;
            let stencil_offset_y_value = stencil_offset_y[y_index];
            for (var offset_x = -1i; offset_x <= 1i; offset_x = offset_x + 1i) {
                let x_index = u32(offset_x + 1i);
                let cell_weight = stencil_weight_x[x_index] * wy_wz;
                let cell_velocity = velocityFromTile(
                    vec3i(start_cell_number.x + offset_x, cell_y, cell_z),
                    local_grid_origin,
                );
                if all(cell_velocity == vec3f(0.0)) { continue; }
                has_velocity_contribution = true;

                let weighted_cell_velocity = cell_weight * cell_velocity;
                new_particle_velocity += weighted_cell_velocity;

                b_matrix += mat3x3f(
                    weighted_cell_velocity * stencil_offset_x[x_index],
                    weighted_cell_velocity * stencil_offset_y_value,
                    weighted_cell_velocity * stencil_offset_z_value,
                );
            }
        }
    }

    let clamped_velocity = clampVec3LengthWithMaxSquared(
        new_particle_velocity,
        maxFixedPointGridSpeed(),
        maxFixedPointGridSpeedSquared(),
    );
    var deformation_displacement = mat3x3f();
    if has_velocity_contribution {
        deformation_displacement = sanitizeVelocityGradient(scaleMatrixColumns(
            b_matrix,
            uniforms.mlsDeformationGradientScale,
        ));
    }

    return FusedGatherResult(clamped_velocity, deformation_displacement, has_velocity_contribution);
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

fn integrateFusedParticle(
    particle: ptr<function, ParticleData>,
    flags: ptr<function, u32>,
) -> f32 {
    var speed_squared = 0.0;
    let material = particleMaterial(*flags);

    applyMaterialVelocityDamping(particle, material);
    let unclamped_velocity = sanitizeVec3((*particle).vel, vec3f(0.0));
    let unclamped_speed_squared = dot(unclamped_velocity, unclamped_velocity);
    (*particle).vel = clampVec3LengthNoSanitizeWithMaxSquared(
        unclamped_velocity,
        uniforms.maxStableParticleSpeed,
        uniforms.maxStableParticleSpeedSquared,
    );

    (*particle).pos_displacement = (*particle).vel * uniforms.simulationTimestep;
    (*particle).pos += (*particle).pos_displacement;

    let deformation_matrices_changed = ((*flags) & PARTICLE_FLAG_DEFORMATION_DELTA_VALID) != 0u;
    var deformation_delta_remains_valid = deformation_matrices_changed;
    var deformation_plastic_changed = false;
    if deformation_matrices_changed {
        let deformation_delta = deformationDeltaFromVelocityGradient((*particle).deformation_displacement);
        if mat3x3IsZero(deformation_delta) {
            (*particle).deformation_displacement = mat3x3f();
            deformation_delta_remains_valid = false;
        } else {
            (*particle).deformationElastic = (IDENTITY_MAT3 + deformation_delta)
            * (*particle).deformationElastic;
        }

        let deformation_det = determinant((*particle).deformationElastic);
        if deformation_det != deformation_det || deformation_det < 0.05 || deformation_det > 20.0 {
            (*particle).deformationElastic = IDENTITY_MAT3;
            (*particle).deformationPlastic = IDENTITY_MAT3;
            (*particle).deformation_displacement = mat3x3f();
            deformation_delta_remains_valid = false;
            deformation_plastic_changed = true;
        } else {
            deformation_plastic_changed = applyPlasticity(particle, material);
        }
    }

    applySimulationDomainBoundary(particle, material);
    resolveParticleCollision(particle);
    sanitizeParticleKinematicsWithoutDeformationDeltaWithKnownScalarRepairs(
        particle,
        false,
        false,
    );

    if deformation_matrices_changed {
        (*particle).deformation_displacement = sanitizeVelocityGradient(
            (*particle).deformation_displacement,
        );
        let matrix_sanitize_flags = sanitizeParticleMatricesAndGetChangedFlags(particle);
        deformation_plastic_changed = deformation_plastic_changed
            || (matrix_sanitize_flags & PARTICLE_MATRIX_PLASTIC_CHANGED) != 0u;
        deformation_delta_remains_valid = !mat3x3IsZero((*particle).deformation_displacement);

        if mat3x3IsIdentity((*particle).deformationElastic) {
            *flags = (*flags) & ~PARTICLE_FLAG_ELASTIC_NON_IDENTITY;
        } else {
            *flags = (*flags) | PARTICLE_FLAG_ELASTIC_NON_IDENTITY;
        }
        if !deformation_delta_remains_valid {
            *flags = (*flags) & ~PARTICLE_FLAG_DEFORMATION_DELTA_VALID;
        }
    }

    if RECORD_PARTICLE_SPEED != 0u {
        speed_squared = unclamped_speed_squared;
    }
    return speed_squared;
}

fn scatterMlsParticleToGrid(
    particle: ptr<function, ParticleData>,
    flags: u32,
    local_grid_origin: vec3i,
) {
    let particle_pos = (*particle).pos;
    let particle_grid_coord = calculateGridCoordinate(particle_pos);
    if !gridCoordinateCanTouchGrid(particle_grid_coord) { return; }

    let start_cell_number = vec3i(floor(particle_grid_coord));
    let cell_frac_pos = particle_grid_coord - vec3f(start_cell_number);
    let cell_weights = calculateQuadraticBSplineCellWeightVectors(cell_frac_pos);
    let stencil_weight_x = cell_weights.x;
    let stencil_weight_y = cell_weights.y;
    let stencil_weight_z = cell_weights.z;
    let particle_mass_fixed_units = (*particle).mass * FIXED_POINT_SCALE;
    let particle_momentum_fixed_units = particle_mass_fixed_units * (*particle).vel;
    let max_particle_momentum_fixed_units = particle_mass_fixed_units * maxFixedPointGridSpeed();
    let has_deformation_delta = (flags & PARTICLE_FLAG_DEFORMATION_DELTA_VALID) != 0u;
    let elastic_is_identity = (flags & PARTICLE_FLAG_ELASTIC_NON_IDENTITY) == 0u;
    let material = particleMaterial(flags);

    if elastic_is_identity && !has_deformation_delta {
        for (var offset_z = -1i; offset_z <= 1i; offset_z = offset_z + 1i) {
            let cell_z = start_cell_number.z + offset_z;
            let z_index = u32(offset_z + 1i);
            let wz = stencil_weight_z[z_index];
            for (var offset_y = -1i; offset_y <= 1i; offset_y = offset_y + 1i) {
                let cell_y = start_cell_number.y + offset_y;
                let y_index = u32(offset_y + 1i);
                let wy_wz = stencil_weight_y[y_index] * wz;
                for (var offset_x = -1i; offset_x <= 1i; offset_x = offset_x + 1i) {
                    let x_index = u32(offset_x + 1i);
                    let cell_weight = stencil_weight_x[x_index] * wy_wz;
                    accumulateFixedUnitsToTileOrNextGrid(
                        vec3i(start_cell_number.x + offset_x, cell_y, cell_z),
                        local_grid_origin,
                        cell_weight * particle_momentum_fixed_units,
                        cell_weight * particle_mass_fixed_units,
                    );
                }
            }
        }
        return;
    }

    var deformation_displacement = mat3x3f();
    if has_deformation_delta {
        deformation_displacement = sanitizeVelocityGradient((*particle).deformation_displacement);
    }

    var stress_force_matrix = mat3x3f();
    if !elastic_is_identity {
        var shear_resistance = SHEAR_RESISTANCE;
        var volumetric_resistance = VOLUME_RESISTANCE;
        applyMaterialLameParameters(material, &shear_resistance, &volumetric_resistance);
        hardenLameParameters(material, (*particle).deformationPlastic, &shear_resistance, &volumetric_resistance);
        let stress = calculateStressFirstPiolaKirchhoffNonIdentity(
            (*particle).deformationElastic,
            shear_resistance,
            volumetric_resistance,
        );
        stress_force_matrix = stress * transpose((*particle).deformationElastic);
    }

    let affine_velocity = deformation_displacement;
    var mls_affine_fixed_units = particle_mass_fixed_units * affine_velocity;
    let grid_cell_dims = uniforms.gridCellDims;
    let stencil_offset_x = (vec3f(-0.5, 0.5, 1.5) - vec3f(cell_frac_pos.x)) * grid_cell_dims.x;
    let stencil_offset_y = (vec3f(-0.5, 0.5, 1.5) - vec3f(cell_frac_pos.y)) * grid_cell_dims.y;
    let stencil_offset_z = (vec3f(-0.5, 0.5, 1.5) - vec3f(cell_frac_pos.z)) * grid_cell_dims.z;
    if !elastic_is_identity {
        let mls_stress_affine_fixed_units = scaleMatrixColumns(
            (*particle).mass * stress_force_matrix,
            uniforms.mlsStressAffineScale,
        );
        mls_affine_fixed_units = mls_affine_fixed_units + mls_stress_affine_fixed_units;
    }

    for (var offset_z = -1i; offset_z <= 1i; offset_z = offset_z + 1i) {
        let cell_z = start_cell_number.z + offset_z;
        let z_index = u32(offset_z + 1i);
        let wz = stencil_weight_z[z_index];
        let mls_affine_z_contribution = mls_affine_fixed_units[2] * stencil_offset_z[z_index];
        for (var offset_y = -1i; offset_y <= 1i; offset_y = offset_y + 1i) {
            let cell_y = start_cell_number.y + offset_y;
            let y_index = u32(offset_y + 1i);
            let wy_wz = stencil_weight_y[y_index] * wz;
            let mls_affine_y_contribution = mls_affine_fixed_units[1] * stencil_offset_y[y_index];
            for (var offset_x = -1i; offset_x <= 1i; offset_x = offset_x + 1i) {
                let x_index = u32(offset_x + 1i);
                let cell_weight = stencil_weight_x[x_index] * wy_wz;
                let weighted_mass_fixed_units = cell_weight * particle_mass_fixed_units;
                let affine_offset_fixed_units = mls_affine_fixed_units[0] * stencil_offset_x[x_index]
                    + mls_affine_y_contribution
                    + mls_affine_z_contribution;
                let momentum_fixed_units = clampMomentumFixedUnitsForAtomic(
                    cell_weight * (particle_momentum_fixed_units + affine_offset_fixed_units),
                    max_particle_momentum_fixed_units,
                );

                accumulateFixedUnitsToTileOrNextGrid(
                    vec3i(start_cell_number.x + offset_x, cell_y, cell_z),
                    local_grid_origin,
                    momentum_fixed_units,
                    weighted_mass_fixed_units,
                );
            }
        }
    }
}

fn flushTileCellToNextGrid(tile_index: u32, local_grid_origin: vec3i) {
    let base_index = tileAccumulatorIndex(tile_index);
    let mass_i = atomicLoad(&tile_accumulator[base_index]);
    let momentum_i = vec3i(
        atomicLoad(&tile_accumulator[base_index + 1u]),
        atomicLoad(&tile_accumulator[base_index + 2u]),
        atomicLoad(&tile_accumulator[base_index + 3u]),
    );
    if mass_i == 0i && all(momentum_i == vec3i(0)) {
        return;
    }

    let cell_number = local_grid_origin + tileOffsetFromIndex(tile_index);
    accumulateFixedUnitsToNextGridCell(cell_number, momentum_i, mass_i);
}

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

@compute
@workgroup_size(PARTICLE_WORKGROUP_SIZE)
fn doFusedMlsG2p2g(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
) {
    let local_index = lid.x;
    let thread_data_index = wid.y * BUKKIT_DISPATCH_WIDTH + wid.x;

    if local_index == 0u {
        workgroup_has_thread_data = thread_data_index < bukkit_thread_group_count.count;
        if workgroup_has_thread_data {
            let thread_data = bukkit_thread_data[thread_data_index];
            workgroup_local_grid_origin = vec3i(
                i32(thread_data.origin_cell_x) - i32(FUSED_BUKKIT_HALO),
                i32(thread_data.origin_cell_y) - i32(FUSED_BUKKIT_HALO),
                i32(thread_data.origin_cell_z) - i32(FUSED_BUKKIT_HALO),
            );
        } else {
            workgroup_local_grid_origin = vec3i(0);
        }
    }
    workgroupBarrier();

    let local_grid_origin = workgroup_local_grid_origin;

    for (var tile_index = local_index; tile_index < FUSED_TILE_CELL_COUNT; tile_index = tile_index + PARTICLE_WORKGROUP_SIZE) {
        var cell_velocity = vec3f(0.0);
        if workgroup_has_thread_data {
            let cell_number = local_grid_origin + tileOffsetFromIndex(tile_index);
            cell_velocity = loadTileVelocity(cell_number);
        }
        tile_velocity[tile_index] = vec4f(cell_velocity, 0.0);
    }

    workgroupBarrier();

    var speed_squared = 0.0;

    if workgroup_has_thread_data {
        let thread_data = bukkit_thread_data[thread_data_index];
        if local_index < thread_data.range_count {
            let particle_index = bukkit_particle_data[thread_data.range_start + local_index];
            if particle_index < N_PARTICLES {
                var particle = particle_data[particle_index];
                var flags = particlePersistentFlags(particle_flags[particle_index]);

                let gather_result = gatherMlsFromTile(particle.pos, local_grid_origin);
                if gather_result.has_velocity_contribution {
                    particle.vel = gather_result.velocity;
                    if mat3x3IsZero(gather_result.deformation_displacement) {
                        flags = flags & ~PARTICLE_FLAG_DEFORMATION_DELTA_VALID;
                    } else {
                        particle.deformation_displacement = gather_result.deformation_displacement;
                        flags = flags | PARTICLE_FLAG_DEFORMATION_DELTA_VALID;
                    }
                } else {
                    flags = flags & ~PARTICLE_FLAG_DEFORMATION_DELTA_VALID;
                }

                speed_squared = integrateFusedParticle(&particle, &flags);
                scatterMlsParticleToGrid(&particle, flags, local_grid_origin);

                particle_data[particle_index].pos = particle.pos;
                particle_data[particle_index].vel = particle.vel;
                particle_data[particle_index].deformation_displacement = particle.deformation_displacement;
                particle_data[particle_index].deformationElastic = particle.deformationElastic;
                particle_data[particle_index].deformationPlastic = particle.deformationPlastic;
                particle_flags[particle_index] = flags;
            }
        }
    }

    workgroupBarrier();

    for (var tile_index = local_index; tile_index < FUSED_TILE_CELL_COUNT; tile_index = tile_index + PARTICLE_WORKGROUP_SIZE) {
        if workgroup_has_thread_data {
            flushTileCellToNextGrid(tile_index, local_grid_origin);
        }
    }

    if RECORD_PARTICLE_SPEED != 0u {
        recordWorkgroupMaxParticleSpeed(local_index, speed_squared);
    }
}
