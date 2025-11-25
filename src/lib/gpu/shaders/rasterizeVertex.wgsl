@group(1) @binding(0)
var<storage, read> colliderVertices : array<f32>;

@group(1) @binding(1)
var<storage, read> colliderIndices : array<u32>;

// const cubePositions : array<vec3<f32>, 8> = array<vec3<f32>, 8>(
//     vec3<f32>(1.0, 0.0, 1.0), // 0
//     vec3<f32>(2.0, 0.0, 1.0), // 1
//     vec3<f32>(2.0, 1.0, 1.0), // 2
//     vec3<f32>(1.0, 1.0, 1.0), // 3
//     vec3<f32>(1.0, 0.0, 2.0), // 4
//     vec3<f32>(2.0, 0.0, 2.0), // 5
//     vec3<f32>(2.0, 1.0, 2.0), // 6
//     vec3<f32>(1.0, 1.0, 2.0)  // 7
// );


// // 12 triangles * 3 indices = 36 verts
// const cubeIndices : array<u32, 36> = array<u32, 36>(
//     // front
//     0, 1, 2,  0, 2, 3,
//     // back
//     5, 4, 7,  5, 7, 6,
//     // left
//     4, 0, 3,  4, 3, 7,
//     // right
//     1, 5, 6,  1, 6, 2,
//     // top
//     3, 2, 6,  3, 6, 7,
//     // bottom
//     4, 5, 1,  4, 1, 0
// );

// const cubeEdges : array<u32, 24> = array<u32,24>(
//     0,1,   1,2,   2,3,   3,0,       // front quad
//     4,5,   5,6,   6,7,   7,4,       // back quad
//     0,4,   1,5,   2,6,   3,7        // connectors
// );

struct VSIn {
    @location(0) position: vec3<f32>,
};

struct VSOut {
    @builtin(position) position : vec4<f32>,
};

@vertex
fn vert(in: VSIn) -> VSOut {
    var out: VSOut;
    out.position = uniforms.viewProjMat * vec4<f32>(in.position, 1.0);
    return out;
}