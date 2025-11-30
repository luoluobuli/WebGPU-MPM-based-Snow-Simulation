@group(1) @binding(0) var<storage, read_write> hash_map_entries: array<HashMapEntry>;
@group(1) @binding(1) var<storage, read_write> n_allocated_blocks: atomic<u32>;
@group(1) @binding(2) var<storage, read_write> mapped_block_indexes: array<u32>; // Stores indices into PageTable
@group(1) @binding(3) var<storage, read_write> grid_mass: array<atomic<i32>>;
@group(1) @binding(4) var<storage, read_write> grid_momentum_x: array<atomic<i32>>;
@group(1) @binding(5) var<storage, read_write> grid_momentum_y: array<atomic<i32>>;
@group(1) @binding(6) var<storage, read_write> grid_momentum_z: array<atomic<i32>>;
@group(1) @binding(7) var<storage, read_write> grid_mass_displacement_x: array<atomic<i32>>;
@group(1) @binding(8) var<storage, read_write> grid_mass_displacement_y: array<atomic<i32>>;
@group(1) @binding(9) var<storage, read_write> grid_mass_displacement_z: array<atomic<i32>>;

fn cubeSDF(p: vec3<f32>, minB: vec3<f32>, maxB: vec3<f32>) -> f32 {
    // distance outside cube
    let outside = max(max(minB - p, p - maxB), vec3<f32>(0.0, 0.0, 0.0));
    let outsideDist = length(outside);

    // negative distance inside cube
    let insideDist = min(
        min(p.x - minB.x, maxB.x - p.x),
        min(p.y - minB.y, maxB.y - p.y)
    );
    
    let insideDist2 = min(insideDist, min(p.z - minB.z, maxB.z - p.z));

    // If outside -> positive, if inside -> negative
    return select(-insideDist2, outsideDist, outsideDist > 0.0);
}

@compute
@workgroup_size(64) // run per block
fn doGridUpdate(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
) {
    let block_index = wid.y * 256 + wid.x;
    if block_index >= N_MAX_BLOCKS_IN_HASH_MAP { return; }
    
    let count = atomicLoad(&n_allocated_blocks);
    if block_index >= count { return; }
    
    let mapped_block_index = mapped_block_indexes[block_index];
    let block_number = hash_map_entries[mapped_block_index].block_number;
    
    let cell_index_within_block = lid.x;
    let cell_index = block_index * 64u + cell_index_within_block;
    
    let cellMass = f32(atomicLoad(&grid_mass[cell_index])) / uniforms.fixedPointScale;
    if cellMass <= 0.0 { return; }

    if uniforms.use_pbmpm == 0 {
        let momX = f32(atomicLoad(&grid_momentum_x[cell_index])) / uniforms.fixedPointScale;
        let momY = f32(atomicLoad(&grid_momentum_y[cell_index])) / uniforms.fixedPointScale;
        let momZ = f32(atomicLoad(&grid_momentum_z[cell_index])) / uniforms.fixedPointScale;

        var v = vec3f(momX, momY, momZ) / cellMass;

        // ------------ Gravity -------------
        let gravity = vec3f(0.0, 0.0, -9.81);
        v += gravity * uniforms.simulationTimestep;
        
        // ----------- Collision ------------
        let minB = (uniforms.colliderTransformMat * vec4f(uniforms.colliderMinCoords, 1.0)).xyz; 
        let maxB = (uniforms.colliderTransformMat * vec4f(uniforms.colliderMaxCoords, 1.0)).xyz;



        let cell_number_within_block_z = i32(cell_index_within_block / 16u);
        let cell_number_within_block_y = i32((cell_index_within_block / 4u) % 4u);
        let cell_number_within_block_x = i32(cell_index_within_block % 4u);
        
        let cell_number = block_number * 4 + vec3i(cell_number_within_block_x, cell_number_within_block_y, cell_number_within_block_z);
        
        let cellDims = calculateCellDims();
        let cellWorldPos = uniforms.gridMinCoords + (vec3f(cell_number) + vec3<f32>(0.5, 0.5, 0.5)) * cellDims;



        let dist = cubeSDF(cellWorldPos, minB, maxB);
        if (dist < 0.05) {
            let projected = clamp(cellWorldPos, minB, maxB);
            var normal = cellWorldPos - projected;
            let nLen = length(normal);

            if (nLen > 1e-6) {
                normal = normal / nLen;

                var v_rel = v - uniforms.colliderVelocity * 100.0;
                let vn = dot(v_rel, normal);

                if (vn < 0.0) {
                    let vN = vn * normal;
                    let vT = v_rel - vN;

                    let friction = 0.3;
                    v_rel = vT * (1.0 - friction);
                }
                v = v_rel + uniforms.colliderVelocity * 100.0; 
            }
            else {
                v = vec3f(0.0, 0.0, 0.0);
            }
        }
        // ----------------------------------

        let newMomentum = v * cellMass * uniforms.fixedPointScale;

        atomicStore(&grid_momentum_x[cell_index], i32(newMomentum.x));
        atomicStore(&grid_momentum_y[cell_index], i32(newMomentum.y));
        atomicStore(&grid_momentum_z[cell_index], i32(newMomentum.z));
    }
    
    else {
        let cell_momentum = vec3f(
            f32(atomicLoad(&grid_momentum_x[cell_index])) / uniforms.fixedPointScale,
            f32(atomicLoad(&grid_momentum_y[cell_index])) / uniforms.fixedPointScale,
            f32(atomicLoad(&grid_momentum_z[cell_index])) / uniforms.fixedPointScale,
        );

        let cell_mass = f32(atomicLoad(&grid_mass[cell_index])) / uniforms.fixedPointScale;

        var cell_velocity = cell_momentum / cell_mass;

        let gravitational_acceleration = vec3f(0, 0, -9.81);
        cell_velocity += gravitational_acceleration * uniforms.simulationTimestep;

        let new_momentum = cell_velocity * cell_mass * uniforms.fixedPointScale;

        atomicStore(&grid_momentum_x[cell_index], i32(new_momentum.x));
        atomicStore(&grid_momentum_y[cell_index], i32(new_momentum.y));
        atomicStore(&grid_momentum_z[cell_index], i32(new_momentum.z));

        // let cell_mass_displacement = vec3f(
        //     f32(atomicLoad(&grid_mass_displacement_x[cell_index])) / uniforms.fixedPointScale,
        //     f32(atomicLoad(&grid_mass_displacement_y[cell_index])) / uniforms.fixedPointScale,
        //     f32(atomicLoad(&grid_mass_displacement_z[cell_index])) / uniforms.fixedPointScale,
        // );

        // let cell_mass = f32(atomicLoad(&grid_mass[cell_index])) / uniforms.fixedPointScale;

        // let cell_displacement = cell_mass_displacement * (1 / cell_mass);
    }
}