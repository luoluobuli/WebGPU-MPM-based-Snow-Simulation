@group(1) @binding(1) var<storage, read_write> grid_momentum_x: array<atomic<i32>>;
@group(1) @binding(2) var<storage, read_write> grid_momentum_y: array<atomic<i32>>;
@group(1) @binding(3) var<storage, read_write> grid_momentum_z: array<atomic<i32>>;
@group(1) @binding(4) var<storage, read_write> grid_mass: array<atomic<i32>>;


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
@workgroup_size(8, 8, 4)
fn doGridUpdate(
    @builtin(global_invocation_id) gid: vec3u,
) {
    if (gid.x >= uniforms.gridResolution.x || gid.y >= uniforms.gridResolution.y || gid.z >= uniforms.gridResolution.z) {
        return;
    }

    let threadIndex = gid.x + uniforms.gridResolution.x * (gid.y + uniforms.gridResolution.y * gid.z);
    if (threadIndex >= arrayLength(&grid_mass)) { return; }

    let cellMass = f32(atomicLoad(&grid_mass[threadIndex])) / uniforms.fixedPointScale;
    if cellMass <= 0.0 { return; }

    let momX = f32(atomicLoad(&grid_momentum_x[threadIndex])) / uniforms.fixedPointScale;
    let momY = f32(atomicLoad(&grid_momentum_y[threadIndex])) / uniforms.fixedPointScale;
    let momZ = f32(atomicLoad(&grid_momentum_z[threadIndex])) / uniforms.fixedPointScale;

    var v = vec3f(momX, momY, momZ) / cellMass;

    // ------------ Gravity -------------
    let gravity = vec3f(0.0, 0.0, -9.81);
    v += gravity * uniforms.simulationTimestep;
    
    // ----------- Collision ------------
    let minB = (uniforms.colliderTransformMat * vec4f(uniforms.colliderMinCoords, 1.0)).xyz; 
    let maxB = (uniforms.colliderTransformMat * vec4f(uniforms.colliderMaxCoords, 1.0)).xyz;

    let cellIdx3d = vec3<u32>(gid.x, gid.y, gid.z);
    let cellDims = calculateCellDims();
    let cellWorldPos = uniforms.gridMinCoords + (vec3<f32>(cellIdx3d) + vec3<f32>(0.5, 0.5, 0.5)) * cellDims;

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

    atomicStore(&grid_momentum_x[threadIndex], i32(newMomentum.x));
    atomicStore(&grid_momentum_y[threadIndex], i32(newMomentum.y));
    atomicStore(&grid_momentum_z[threadIndex], i32(newMomentum.z));
}