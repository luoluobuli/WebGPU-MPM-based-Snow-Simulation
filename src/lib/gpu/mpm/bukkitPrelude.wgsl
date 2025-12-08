// Bukkit-based spatial partitioning prelude
// Replaces hash map lookups with O(1) direct grid indexing

// Bukkit constants - 4x4x4 cells per bukkit
const BUKKIT_SIZE = 4u;
const BUKKIT_HALO_SIZE = 1u;
const TOTAL_BUKKIT_EDGE = BUKKIT_SIZE + BUKKIT_HALO_SIZE * 2u; // 6
const TILE_DATA_SIZE = TOTAL_BUKKIT_EDGE * TOTAL_BUKKIT_EDGE * TOTAL_BUKKIT_EDGE * 4u; // 6^3 * 4 channels = 864

const PARTICLE_DISPATCH_SIZE = 64u;

struct BukkitThreadData {
    rangeStart: u32,
    rangeCount: u32,
    bukkitX: u32,
    bukkitY: u32,
    bukkitZ: u32,
    _pad: u32,
}

struct BukkitParams {
    countX: u32,
    countY: u32,
    countZ: u32,
    particleCount: u32,
}

fn positionToBukkitId(pos: vec3f) -> vec3i {
    let gridPos = calculateCellNumber(pos);
    return gridPos >> vec3u(2u); // Divide by 4 (BUKKIT_SIZE)
}

fn bukkitIdToIndex(bukkitId: vec3u, countX: u32, countY: u32) -> u32 {
    return bukkitId.z * countX * countY + bukkitId.y * countX + bukkitId.x;
}

fn cellToGridIndex(cellNumber: vec3i) -> u32 {
    // Direct O(1) indexing - no hash map!
    if (!cellNumberInGridRange(cellNumber)) { return 0xFFFFFFFFu; }
    let c = vec3u(cellNumber);
    return c.z * uniforms.gridResolution.x * uniforms.gridResolution.y 
         + c.y * uniforms.gridResolution.x 
         + c.x;
}

fn localGridIndex(localIndex: vec3u) -> u32 {
    return (localIndex.z * TOTAL_BUKKIT_EDGE * TOTAL_BUKKIT_EDGE
          + localIndex.y * TOTAL_BUKKIT_EDGE 
          + localIndex.x) * 4u;  // 4 channels: mass, momX, momY, momZ
}

fn decodeFixedPoint(fixedPoint: i32, scale: f32) -> f32 {
    return f32(fixedPoint) / scale;
}

fn encodeFixedPoint(floatingPoint: f32, scale: f32) -> i32 {
    return i32(floatingPoint * scale);
}
