@group(0) @binding(1) var depthTexture: texture_depth_2d;
@group(0) @binding(2) var smoothedDepthTexture: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var maskTexture: texture_2d<f32>;

const FILTER_RADIUS: i32 = 20;
const STDDEV_SPATIAL: f32 = 5.;
const DEPTH_THRESHOLD: f32 = 0.06;

fn gaussian(x: f32, stddev: f32) -> f32 {
    return exp(-x * x / (2 * stddev * stddev));
}

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let screen_size = textureDimensions(depthTexture);
    let coords = vec2i(global_id.xy);

    if any(coords >= vec2i(screen_size)) { return; }

    let center_mask = textureLoad(maskTexture, coords, 0);
    let center_depth = center_mask.r;
    let center_compression = center_mask.g;

    // valid particle pixel has depth < 1
    if center_depth >= 1 {
        textureStore(smoothedDepthTexture, coords, vec4f(1, 0, 0, 0));
        return;
    }
    
    var total_weight = 0.;
    var total_depth = 0.;
    var total_compression = 0.;

    for (var y = -FILTER_RADIUS; y <= FILTER_RADIUS; y++) {
        for (var x = -FILTER_RADIUS; x <= FILTER_RADIUS; x++) {
            let neighbor_coords = coords + vec2i(x, y);

            if any(neighbor_coords < vec2i(0)) || any(neighbor_coords >= vec2i(screen_size)) {
                continue;
            }

            let neighbor_mask = textureLoad(maskTexture, neighbor_coords, 0);
            let neighbor_depth = neighbor_mask.r;
            let neighbor_compression = neighbor_mask.g;
            
            // skip bg
            if neighbor_depth >= 1 { continue; }

            if abs(center_depth - neighbor_depth) > DEPTH_THRESHOLD { continue; }

            let dist = length(vec2f(f32(x), f32(y)));
            let spatial_weight = gaussian(dist, STDDEV_SPATIAL);

            total_weight += spatial_weight;
            total_depth += spatial_weight * neighbor_depth;
            total_compression += spatial_weight * neighbor_compression;
        }
    }

    let smoothed_depth = select(
        center_depth,
        total_depth / total_weight,
        total_weight > 0,
    );

    let smoothed_compression = select(
        center_compression,
        total_compression / total_weight,
        total_weight > 0,
    );

    textureStore(smoothedDepthTexture, coords, vec4f(smoothed_depth, smoothed_compression, 0, 0));
}
