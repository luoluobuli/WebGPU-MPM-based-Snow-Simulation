@group(0) @binding(0) var depthTex: texture_depth_2d;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> coords: vec2<u32>;

@compute @workgroup_size(1)
fn main() {
    output[0] = textureLoad(depthTex, vec2<i32>(coords), 0);
}
