@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var depthTexture: texture_depth_2d;
@group(0) @binding(2) var smoothedDepthTexture: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var maskTexture: texture_2d<f32>;

const WORLD_STDDEV = 0.1;
const WORLD_FILTER_RADIUS = WORLD_STDDEV * 3;
const DEPTH_WEIGHT_FAC = 90.;
const MAX_SCREEN_RADIUS = 32;

const PROJ_FACTOR = 0.866;

fn gaussian(x: f32, stddev: f32) -> f32 {
    return exp(-x * x / (2 * stddev * stddev));
}

fn linearizeDepth(ndc_depth: f32) -> f32 {
    let near = 0.1;
    let far = 100.;
    return near * far / (far - ndc_depth * (far - near));
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
    
    // Convert to linear depth for world-space calculations
    let linear_depth = linearizeDepth(center_depth);
    
    // Convert world-space radius to screen-space pixels using fixed projection scale
    let proj_scale = f32(screen_size.y) * PROJ_FACTOR;
    let screen_filter_radius = i32(min(WORLD_FILTER_RADIUS * proj_scale / linear_depth, f32(MAX_SCREEN_RADIUS)));
    let screen_stddev = max(WORLD_STDDEV * proj_scale / linear_depth, 1.0);
    
    var total_weight = 0.;
    var total_depth = 0.;
    var total_compression = 0.;

    for (var y = -screen_filter_radius; y <= screen_filter_radius; y++) {
        for (var x = -screen_filter_radius; x <= screen_filter_radius; x++) {
            let neighbor_coords = coords + vec2i(x, y);

            if any(neighbor_coords < vec2i(0)) || any(neighbor_coords >= vec2i(screen_size)) { continue; }

            let neighbor_mask = textureLoad(maskTexture, neighbor_coords, 0);
            let neighbor_depth = neighbor_mask.r;
            let neighbor_compression = neighbor_mask.g;
            
            // skip bg
            if neighbor_depth >= 1 { continue; }

            // Compare linear depths for world-space consistent edge detection
            let neighbor_linear_depth = linearizeDepth(neighbor_depth);
            let depth_diff = (linear_depth - neighbor_linear_depth) / linear_depth; // relative difference
            let depth_weight = exp(-depth_diff * depth_diff * DEPTH_WEIGHT_FAC);

            let dist = length(vec2f(f32(x), f32(y)));
            let spatial_weight = gaussian(dist, screen_stddev);

            let weight = spatial_weight * depth_weight;

            total_weight += weight;
            total_depth += weight * neighbor_depth;
            total_compression += weight * neighbor_compression;
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
