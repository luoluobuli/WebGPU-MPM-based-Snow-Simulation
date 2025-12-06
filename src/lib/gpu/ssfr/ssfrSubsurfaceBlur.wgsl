@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var inputTexture: texture_2d<f32>;
@group(0) @binding(2) var thicknessTexture: texture_2d<f32>;
@group(0) @binding(3) var smoothedDepthTexture: texture_2d<f32>;
@group(0) @binding(4) var outputTexture: texture_storage_2d<rgba8unorm, write>;

const WORLD_BLUR_RADIUS = 0.4;
const MAX_SCREEN_RADIUS = 24;
const DEPTH_WEIGHT_FAC = 90.;
const PROJ_FACTOR = 0.866;

const REFLECTED_COLOR = vec3f(0.4, 0.9, 1);

fn gaussian(x: f32, stddev: f32) -> f32 {
    return exp(-x * x / (2 * stddev * stddev));
}

fn linearizeDepth(ndc_depth: f32) -> f32 {
    let near = 0.1;
    let far = 100.0;
    return near * far / (far - ndc_depth * (far - near));
}

@compute
@workgroup_size(8, 8)
fn mainHorizontal(@builtin(global_invocation_id) global_id: vec3u) {
    let screen_size = vec2i(textureDimensions(inputTexture));
    let coords = vec2i(global_id.xy);
    
    if any(coords >= screen_size) { return; }
    
    let depth_data = textureLoad(smoothedDepthTexture, coords, 0);
    let center_depth = depth_data.r;
    let compression_volume_fac = depth_data.g;
    
    if center_depth >= 1 {
        // bg pixel
        textureStore(outputTexture, coords, vec4f());
        return;
    }
    
    let thickness = textureLoad(thicknessTexture, coords, 0).r;
    let linear_depth = linearizeDepth(center_depth);
    
    // Fixed projection scale
    let proj_scale = f32(screen_size.y) * PROJ_FACTOR;
    
    // thickness modulation: thin edges blur less
    let thickness_factor = saturate(thickness * 2);
    
    // compression modulation: ice scatters less, powder scatters more
    let compression_factor = mix(0.3, 1.0, saturate(compression_volume_fac));
    
    // Calculate world-space blur radius with modulation, then convert to screen-space
    let world_radius = WORLD_BLUR_RADIUS * thickness_factor * compression_factor;
    let screen_stddev = max(world_radius * proj_scale / linear_depth, 0.5);
    let blur_radius = i32(min(screen_stddev * 2, f32(MAX_SCREEN_RADIUS)));
    
    var total_color = vec3f();
    var total_weight = 0.;
    
    for (var i = -blur_radius; i <= blur_radius; i++) {
        let sample_coords = coords + vec2i(i, 0);
        
        if sample_coords.x < 0 || sample_coords.x >= screen_size.x { continue; }
        
        let sample_depth = textureLoad(smoothedDepthTexture, sample_coords, 0).r;
        
        // skip bg
        if sample_depth >= 1 { continue; }

        // use relative linear depth difference for world-space consistency
        let sample_linear_depth = linearizeDepth(sample_depth);
        let depth_diff = (linear_depth - sample_linear_depth) / linear_depth;
        let depth_weight = exp(-depth_diff * depth_diff * DEPTH_WEIGHT_FAC);
        
        let spatial_weight = gaussian(f32(i), max(screen_stddev, 0.5));

        let weight = spatial_weight * depth_weight;


        let sample_color = textureLoad(inputTexture, sample_coords, 0).rgb;
        
        total_color += sample_color * weight;
        total_weight += weight;
    }
    
    var blurred_color = vec3f(0.0);
    if total_weight > 0.0 {
        blurred_color = total_color / total_weight;
    } else {
        blurred_color = textureLoad(inputTexture, coords, 0).rgb;
    }
    
    textureStore(outputTexture, coords, vec4f(blurred_color, 1.0));
}

@compute
@workgroup_size(8, 8)
fn mainVertical(@builtin(global_invocation_id) global_id: vec3u) {
    let screen_size = vec2i(textureDimensions(inputTexture));
    let coords = vec2i(global_id.xy);
    
    if any(coords >= screen_size) { return; }
    
    let depth_data = textureLoad(smoothedDepthTexture, coords, 0);
    let center_depth = depth_data.r;
    let compression_volume_fac = depth_data.g;
    
    if center_depth >= 1 {
        // bg pixel
        textureStore(outputTexture, coords, vec4f());
        return;
    }
    
    let thickness = textureLoad(thicknessTexture, coords, 0).r;
    let linear_depth = linearizeDepth(center_depth);
    
    // Fixed projection scale
    let proj_scale = f32(screen_size.y) * PROJ_FACTOR;
    
    let thickness_factor = saturate(thickness * 2);
    let compression_factor = mix(0.3, 1, saturate(compression_volume_fac));
    
    // Calculate world-space blur radius with modulation, then convert to screen-space
    let world_radius = WORLD_BLUR_RADIUS * thickness_factor * compression_factor;
    let screen_stddev = max(world_radius * proj_scale / linear_depth, 0.5);
    let blur_radius = i32(min(screen_stddev * 2, f32(MAX_SCREEN_RADIUS)));
    
    var total_color = vec3f();
    var total_weight = 0.;
    
    for (var i = -blur_radius; i <= blur_radius; i++) {
        let sample_coords = coords + vec2i(0, i);
        
        if sample_coords.y < 0 || sample_coords.y >= screen_size.y { continue; }
        
        let sample_depth = textureLoad(smoothedDepthTexture, sample_coords, 0).r;

        // Use relative linear depth difference for world-space consistency
        let sample_linear_depth = linearizeDepth(sample_depth);
        let depth_diff = (linear_depth - sample_linear_depth) / linear_depth;
        let depth_weight = exp(-depth_diff * depth_diff * DEPTH_WEIGHT_FAC);
        
        if sample_depth >= 1 { continue; }
        
        let spatial_weight = gaussian(f32(i), max(screen_stddev, 0.5));
        let sample_color = textureLoad(inputTexture, sample_coords, 0).rgb;

        let weight = spatial_weight * depth_weight;
        
        total_color += sample_color * weight;
        total_weight += weight;
    }
    
    var blurred_color = vec3f();
    if total_weight > 0 {
        blurred_color = total_color / total_weight;
    } else {
        blurred_color = textureLoad(inputTexture, coords, 0).rgb;
    }
    
    blurred_color *= REFLECTED_COLOR;
    
    textureStore(outputTexture, coords, vec4f(blurred_color, 1));
}
