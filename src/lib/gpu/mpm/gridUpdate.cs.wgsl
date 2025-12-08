@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Dense grid bind group
@group(1) @binding(3) var<storage, read_write> grid_mass: array<i32>;
@group(1) @binding(4) var<storage, read_write> grid_momentum_x: array<i32>;
@group(1) @binding(5) var<storage, read_write> grid_momentum_y: array<i32>;
@group(1) @binding(6) var<storage, read_write> grid_momentum_z: array<i32>;


@compute
@workgroup_size(256)
fn doGridUpdate(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let totalCells = uniforms.gridResolution.x * uniforms.gridResolution.y * uniforms.gridResolution.z;
    let cell_index = gid.x;
    if cell_index >= totalCells { return; }
    
    let cellMass = f32(grid_mass[cell_index]) / uniforms.fixedPointScale;
    if cellMass <= 0.0 { return; }

    // Reconstruct cell coordinates from linear index
    let resX = uniforms.gridResolution.x;
    let resY = uniforms.gridResolution.y;
    let cell_z = cell_index / (resX * resY);
    let cell_y = (cell_index % (resX * resY)) / resX;
    let cell_x = cell_index % resX;
    let grid_node_pos = vec3f(f32(cell_x), f32(cell_y), f32(cell_z));

    if uniforms.use_pbmpm == 0 {
        let momX = f32(grid_momentum_x[cell_index]) / uniforms.fixedPointScale;
        let momY = f32(grid_momentum_y[cell_index]) / uniforms.fixedPointScale;
        let momZ = f32(grid_momentum_z[cell_index]) / uniforms.fixedPointScale;

        var v = vec3f(momX, momY, momZ) / cellMass;

        // ------------ Gravity -------------
        let gravity = vec3f(0.0, 0.0, -9.81);
        v += gravity * uniforms.simulationTimestep;

        let newMomentum = v * cellMass * uniforms.fixedPointScale;

        grid_momentum_x[cell_index] = i32(newMomentum.x);
        grid_momentum_y[cell_index] = i32(newMomentum.y);
        grid_momentum_z[cell_index] = i32(newMomentum.z);
    }
    
    else {
        let cell_momentum = vec3f(
            f32(grid_momentum_x[cell_index]) / uniforms.fixedPointScale,
            f32(grid_momentum_y[cell_index]) / uniforms.fixedPointScale,
            f32(grid_momentum_z[cell_index]) / uniforms.fixedPointScale,
        );

        let cell_mass = f32(grid_mass[cell_index]) / uniforms.fixedPointScale;

        var cell_velocity = cell_momentum / cell_mass;

        let gravitational_acceleration = vec3f(0, 0, -9.81) / 3;
        cell_velocity += gravitational_acceleration * uniforms.simulationTimestep;

        // Interaction
        if (uniforms.isInteracting != 0u) {
            let dist = distance(grid_node_pos, uniforms.interactionPos);
            if (dist < uniforms.interactionRadius) {
                let offset = grid_node_pos - uniforms.interactionPos;
                var dir = vec3f(0.0, 0.0, 1.0);
                if (length(offset) > 0.001) {
                    dir = normalize(offset);
                }
                
                let falloff = 1.0 - (dist / uniforms.interactionRadius);
                var accel = vec3f(0.0);

                // 1. Attract
                if (uniforms.interactionMode == 1u) {
                     accel = -dir * uniforms.interactionStrength * falloff / 3.0;
                }
                // 0. Repel (Default)
                else {
                    accel = dir * uniforms.interactionStrength * falloff / 3.0; 
                }

                cell_velocity += accel * uniforms.simulationTimestep;
            }
        }

        let new_momentum = cell_velocity * cell_mass * uniforms.fixedPointScale;

        grid_momentum_x[cell_index] = i32(new_momentum.x);
        grid_momentum_y[cell_index] = i32(new_momentum.y);
        grid_momentum_z[cell_index] = i32(new_momentum.z);
    }
}