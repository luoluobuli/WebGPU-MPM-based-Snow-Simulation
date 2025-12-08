struct MCParams {
    mcGridRes: vec3u,
    downsampleFactor: u32,
    densityGridRes: vec3u,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read> particleData: array<ParticleData>;
@group(1) @binding(1) var<storage, read_write> densityGrid: array<atomic<u32>>;
@group(1) @binding(2) var<uniform> mcParams: MCParams;

@compute
@workgroup_size(256)
fn calculateDensity(@builtin(global_invocation_id) global_id: vec3u) {
    let particleIndex = global_id.x;
    if particleIndex >= arrayLength(&particleData) { return; }

    let particle = particleData[particleIndex];
    let pos = particle.pos;

    if any(pos < uniforms.gridMinCoords) || any(pos >= uniforms.gridMaxCoords) { return; }

    let gridRange = uniforms.gridMaxCoords - uniforms.gridMinCoords;
    let gridRes = vec3f(mcParams.densityGridRes);
    let cellSize = gridRange / gridRes;
    
    let posFromGridMin = pos - uniforms.gridMinCoords;
    let posCell = posFromGridMin / cellSize;
    
    let posCellCenter = posCell - 0.5;
    let startCellNumber = vec3i(floor(posCellCenter));
    let fractionalPos = posCellCenter - vec3f(startCellNumber);
    
    let w0 = 1 - fractionalPos;
    let w1 = fractionalPos;

    for (var z = 0; z < 2; z++) {
        for (var y = 0; y < 2; y++) {
            for (var x = 0; x < 2; x++) {
                workgroupBarrier();
                
                let cellNumber = startCellNumber + vec3i(x, y, z);

                if any(cellNumber < vec3i(0)) || any(cellNumber >= vec3i(mcParams.densityGridRes)) { continue; }
                
                let wx = select(w0.x, w1.x, x == 1);
                let wy = select(w0.y, w1.y, y == 1);
                let wz = select(w0.z, w1.z, z == 1);
                let weight = wx * wy * wz;

                let cellIndex = cellNumber.x 
                    + cellNumber.y * i32(mcParams.densityGridRes.x)
                    + cellNumber.z * i32(mcParams.densityGridRes.x * mcParams.densityGridRes.y);

                // Use particle mass as density contribution
                let densityContribution = u32(particle.mass * weight * uniforms.fixedPointScale);
                atomicAdd(&densityGrid[cellIndex], densityContribution);
            }
        }
    }
}

