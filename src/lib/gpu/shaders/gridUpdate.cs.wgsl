@group(1) @binding(1) var<storage, read_write> gridData: array<CellData>;

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
    if (gid.x >= uniforms.gridResolution || gid.y >= uniforms.gridResolution || gid.z >= uniforms.gridResolution) {
        return;
    }

    let threadIndex = gid.x + uniforms.gridResolution * (gid.y + uniforms.gridResolution * gid.z);
    if (threadIndex >= arrayLength(&gridData)) { return; }

    let grid = &gridData[threadIndex];

    let cellMass = f32(atomicLoad(&(*grid).mass)) / uniforms.fixedPointScale;
    if cellMass <= 0.0 { return; }

    let momX = f32(atomicLoad(&(*grid).momentumX)) / uniforms.fixedPointScale;
    let momY = f32(atomicLoad(&(*grid).momentumY)) / uniforms.fixedPointScale;
    let momZ = f32(atomicLoad(&(*grid).momentumZ)) / uniforms.fixedPointScale;

    var v = vec3f(momX, momY, momZ) / cellMass;

    // ------------ Gravity -------------
    let gravity = vec3f(0.0, 0.0, -9.81);
    v = v + gravity * uniforms.simulationTimestep;
    
    // ----------- Collision ------------
    // Temp cube
    let minB = vec3f(1.0, 0.0, 1.0);
    let maxB = vec3f(2.0, 1.0, 2.0);

    // Get grid world pos
    let cellIdx3d = vec3<u32>(gid.x, gid.y, gid.z);

    let cellDims = calculateCellDims();
    let cellWorldPos = uniforms.gridMinCoords + (vec3<f32>(cellIdx3d) + vec3<f32>(0.5, 0.5, 0.5)) * cellDims;

    // Check if object is inside the collider
    let dist = cubeSDF(cellWorldPos, minB, maxB);
    if (dist < 0.0) {
        let projected = clamp(cellWorldPos, minB, maxB);
        var normal = cellWorldPos - projected;
        let nLen = length(normal);

        if (nLen > 1e-6) {
            normal = normal / nLen;

            let vn = dot(v, normal);

            if (vn < 0.0) {
                let vN = vn * normal;
                let vT = v - vN;

                let friction = 0.3;
                v = vT * (1.0 - friction);
            }
        } else {
            // Degenerate normal (exactly at center): just zero velocity
            v = vec3f(0.0, 0.0, 0.0);
        }
    }
    // ----------------------------------

    let newMomentum = v * cellMass * uniforms.fixedPointScale;

    atomicStore(&(*grid).momentumX, i32(newMomentum.x));
    atomicStore(&(*grid).momentumY, i32(newMomentum.y));
    atomicStore(&(*grid).momentumZ, i32(newMomentum.z));
}