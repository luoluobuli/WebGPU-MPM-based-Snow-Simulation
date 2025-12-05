// Marching Cubes Prelude
// Edge numbering scheme:
//        +---------e3--------+
//       /|                  /|
//      / |                 / |
//    e6  |                e7 |
//    /   |               /   |
//   +---------e2--------+    |
//   |    e10            |    e11
//   |    |              |    |
//   |    +---------e1--------+
//   e8  /               e9  /           z
//   |  e4               |  e5           ^  y
//   | /                 | /             | /
//   |/                  |/              +----> x
//   +---------e0--------+
//
// Vertex numbering: 0=origin, +x=1, +y=4, +z=2 bits

// Triangle counts for each of 256 cases
const MC_TRI_COUNTS = array<u32, 256>(
    0u, 1u, 1u, 2u, 1u, 2u, 4u, 3u, 1u, 4u, 2u, 3u, 2u, 3u, 3u, 2u,
    1u, 2u, 4u, 3u, 4u, 3u, 3u, 4u, 2u, 3u, 3u, 4u, 3u, 4u, 4u, 3u,
    1u, 4u, 2u, 3u, 2u, 3u, 3u, 4u, 4u, 3u, 3u, 4u, 3u, 4u, 4u, 3u,
    2u, 3u, 3u, 2u, 3u, 4u, 4u, 3u, 3u, 4u, 4u, 3u, 4u, 3u, 3u, 2u,
    1u, 4u, 2u, 3u, 2u, 3u, 3u, 4u, 4u, 3u, 3u, 4u, 3u, 4u, 4u, 3u,
    2u, 3u, 3u, 4u, 3u, 2u, 4u, 3u, 3u, 4u, 4u, 3u, 4u, 3u, 3u, 2u,
    4u, 3u, 3u, 4u, 3u, 4u, 4u, 3u, 3u, 4u, 4u, 3u, 4u, 3u, 3u, 4u,
    3u, 4u, 4u, 3u, 4u, 3u, 3u, 2u, 4u, 3u, 3u, 4u, 3u, 4u, 2u, 1u,
    1u, 2u, 4u, 3u, 4u, 3u, 3u, 4u, 2u, 3u, 3u, 4u, 3u, 4u, 4u, 3u,
    4u, 3u, 3u, 4u, 3u, 4u, 4u, 3u, 3u, 4u, 4u, 3u, 4u, 3u, 3u, 4u,
    2u, 3u, 3u, 4u, 3u, 4u, 4u, 3u, 3u, 4u, 2u, 3u, 4u, 3u, 3u, 2u,
    3u, 4u, 4u, 3u, 4u, 3u, 3u, 4u, 4u, 3u, 3u, 2u, 3u, 2u, 4u, 1u,
    2u, 3u, 3u, 4u, 3u, 4u, 4u, 3u, 3u, 4u, 4u, 3u, 2u, 3u, 3u, 2u,
    3u, 4u, 4u, 3u, 4u, 3u, 3u, 4u, 4u, 3u, 3u, 2u, 3u, 2u, 4u, 1u,
    3u, 4u, 4u, 3u, 4u, 3u, 3u, 2u, 4u, 3u, 3u, 4u, 3u, 4u, 2u, 1u,
    2u, 3u, 3u, 2u, 3u, 2u, 4u, 1u, 3u, 4u, 2u, 1u, 2u, 1u, 1u, 0u
);

// Edge endpoints - each edge connects two vertices
// Using vertex numbering: bit0=x, bit1=z, bit2=y
// v0=(0,0,0), v1=(1,0,0), v2=(0,0,1), v3=(1,0,1), v4=(0,1,0), v5=(1,1,0), v6=(0,1,1), v7=(1,1,1)
//
// Edge numbering (Verified by reverse-engineering table):
// X-axis: e0(v0-v1), e1(v4-v5), e2(v2-v3), e3(v6-v7)  <-- Note e1/e2 are swapped vs standard
// Y-axis: e4(v0-v4), e5(v1-v5), e6(v2-v6), e7(v3-v7)
// Z-axis: e8(v0-v2), e9(v1-v3), e10(v4-v6), e11(v5-v7)
const MC_EDGE_V0 = array<u32, 12>(0u, 4u, 2u, 6u, 0u, 1u, 2u, 3u, 0u, 1u, 4u, 5u);
const MC_EDGE_V1 = array<u32, 12>(1u, 5u, 3u, 7u, 4u, 5u, 6u, 7u, 2u, 3u, 6u, 7u);

// Vertex offsets - convert vertex index to 3D offset
fn mcVertexOffset(v: u32) -> vec3f {
    return vec3f(
        f32(v & 1u),
        f32((v >> 2u) & 1u),
        f32((v >> 1u) & 1u)
    );
}

// Get edge endpoints as 3D positions relative to cell origin
fn mcEdgeEndpoints(edge: u32) -> array<vec3f, 2> {
    return array<vec3f, 2>(
        mcVertexOffset(MC_EDGE_V0[edge]),
        mcVertexOffset(MC_EDGE_V1[edge])
    );
}

// Interpolate along edge based on density values
fn mcInterpolateEdge(edge: u32, densities: array<f32, 8>, isovalue: f32) -> vec3f {
    let v0 = MC_EDGE_V0[edge];
    let v1 = MC_EDGE_V1[edge];
    let d0 = densities[v0];
    let d1 = densities[v1];
    let t = clamp((isovalue - d0) / (d1 - d0 + 0.0001), 0.0, 1.0);
    return mix(mcVertexOffset(v0), mcVertexOffset(v1), t);
}

// Compute marching cubes case from 8 corner densities
fn mcComputeCase(densities: array<f32, 8>, isovalue: f32) -> u32 {
    var mcCase = 0u;
    for (var i = 0u; i < 8u; i++) {
        if densities[i] > isovalue {
            mcCase |= (1u << i);
        }
    }
    return mcCase;
}
