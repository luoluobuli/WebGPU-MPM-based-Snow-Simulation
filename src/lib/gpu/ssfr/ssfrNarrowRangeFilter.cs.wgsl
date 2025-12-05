@group(0) @binding(1) var depthTexture: texture_depth_2d;
@group(0) @binding(2) var smoothedDepthTexture: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var maskTexture: texture_2d<f32>;

const FILTER_RADIUS: i32 = 20;
const SIGMA_SPATIAL: f32 = 5.0;
const DEPTH_THRESHOLD: f32 = 0.01;

fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-x * x / (2 * sigma * sigma));
}

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let screen_size = textureDimensions(depthTexture);
    let coords = vec2i(global_id.xy);

    if any(coords >= vec2i(screen_size)) { return; }

    let center_depth = textureLoad(maskTexture, coords, 0).r;

    // valid particle pixel has depth < 1
    if center_depth >= 1 {
        textureStore(smoothedDepthTexture, coords, vec4f(1.0, 0.0, 0.0, 0.0));
        return;
    }
    
    var total_weight = 0.;
    var total_depth = 0.;

    for (var y = -FILTER_RADIUS; y <= FILTER_RADIUS; y++) {
        for (var x = -FILTER_RADIUS; x <= FILTER_RADIUS; x++) {
            let neighbor_coords = coords + vec2i(x, y);

            if any(neighbor_coords < vec2i(0)) || any(neighbor_coords >= vec2i(screen_size)) {
                continue;
            }

            let neighbor_depth = textureLoad(maskTexture, neighbor_coords, 0).r;
            
            // skip bg
            if neighbor_depth >= 1 { continue; }

            if abs(center_depth - neighbor_depth) > DEPTH_THRESHOLD { continue; }

            let dist = length(vec2f(f32(x), f32(y)));
            let spatial_weight = gaussian(dist, SIGMA_SPATIAL);

            total_weight += spatial_weight;
            total_depth += spatial_weight * neighbor_depth;
        }
    }

    var smoothed_depth = center_depth;
    if total_weight > 0.0 {
        smoothed_depth = total_depth / total_weight;
    }

    textureStore(smoothedDepthTexture, coords, vec4f(smoothed_depth, 0.0, 0.0, 0.0));
}
