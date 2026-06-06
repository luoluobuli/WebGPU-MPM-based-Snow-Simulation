@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct FragmentInput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) center_world: vec3f,
    @location(2) radius: f32,
    @location(3) compression_volume_fac: f32,
    @location(4) elastic_volume_fac: f32,
    @location(5) shape_params: vec4f,
    @location(6) grain_params: vec4f,
    @location(7) appearance: vec4f,
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

const LIGHT_DIR = vec3f(0.45, 0.32, 0.84);
const AMBIENT_COLOR = vec3f(0.19, 0.22, 0.28);
const SNOW_ALBEDO = vec3f(0.88, 0.93, 0.98);
const SKY_FILL = vec3f(0.12, 0.16, 0.23);
const NOISE_SCALE = 24.0;
const SILHOUETTE_NOISE_SCALE = 5.5;

fn cameraBillboardBasis(center_world: vec3f) -> mat3x3f {
    let forward = normalize(uniforms.cameraPos - center_world);
    let reference_up = select(vec3f(0, 0, 1), vec3f(0, 1, 0), abs(forward.z) > 0.95);
    let right = normalize(cross(reference_up, forward));
    let up = cross(forward, right);

    return mat3x3f(right, up, forward);
}

fn noiseGradient3(p: vec3f) -> vec3f {
    let eps = 0.01;
    let n = noise3(p);
    let nx = noise3(p + vec3f(eps, 0, 0));
    let ny = noise3(p + vec3f(0, eps, 0));
    let nz = noise3(p + vec3f(0, 0, eps));

    return vec3f(nx - n, ny - n, nz - n) / eps;
}

fn perturbSnowNormal(local_pos: vec3f, normal_world: vec3f, loose_factor: f32, seed: vec4f) -> vec3f {
    let grain_gradient = noiseGradient3(local_pos * NOISE_SCALE + seed.xyz * 17.0);
    let tangent_gradient = grain_gradient - dot(grain_gradient, normal_world) * normal_world;
    let noise_strength = mix(0.07, 0.16, loose_factor);

    return normalize(normal_world + tangent_gradient * noise_strength);
}

fn shadeSnow(pos_world: vec3f, local_pos: vec3f, normal_world: vec3f, loose_factor: f32, seed: vec4f) -> vec3f {
    let normal = perturbSnowNormal(local_pos, normal_world, loose_factor, seed);
    let view_dir = normalize(uniforms.cameraPos - pos_world);
    let half_dir = normalize(LIGHT_DIR + view_dir);

    let diffuse = max(dot(normal, LIGHT_DIR), 0.0);
    let wrap_diffuse = diffuse * 0.62 + 0.38;
    let sky = max(normal.z * 0.5 + 0.5, 0.0);
    let fresnel = pow(1.0 - saturate(dot(normal, view_dir)), 4.0);

    let grain = noise3(local_pos * 58.0 + seed.yzx * 23.0);
    let grain_tint = mix(0.84, 1.09, grain);
    let specular = pow(max(dot(normal, half_dir), 0.0), 24.0) * 0.055;
    let glint_cell = floor(local_pos * mix(14.0, 22.0, loose_factor) + seed.zxy * 31.0);
    let glint_seed = hash31(glint_cell + vec3f(13.0, 37.0, 71.0));
    let glint = smoothstep(0.985, 1.0, glint_seed) * pow(max(dot(normal, half_dir), 0.0), 6.0);

    let diffuse_color = SNOW_ALBEDO * wrap_diffuse * grain_tint;
    let ambient = AMBIENT_COLOR + SKY_FILL * sky;

    return diffuse_color + ambient + vec3f(specular + glint * 0.18) + fresnel * vec3f(0.025, 0.035, 0.05);
}

fn shadeForestParticle(
    pos_world: vec3f,
    local_pos: vec3f,
    normal_world: vec3f,
    appearance: vec4f,
    seed: vec4f,
) -> vec3f {
    let grain_gradient = noiseGradient3(local_pos * 20.0 + seed.xyz * 13.0);
    let tangent_gradient = grain_gradient - dot(grain_gradient, normal_world) * normal_world;
    let normal = normalize(normal_world + tangent_gradient * 0.14);
    let view_dir = normalize(uniforms.cameraPos - pos_world);
    let half_dir = normalize(LIGHT_DIR + view_dir);
    let diffuse = max(dot(normal, LIGHT_DIR), 0.0);
    let sky = max(normal.z * 0.5 + 0.5, 0.0);
    let material = appearance.w;
    var albedo = appearance.rgb;

    if material > 2.5 {
        let leaf_mottle = noise3(pos_world * 6.0 + seed.zxy * 23.0);
        albedo *= mix(0.72, 1.22, leaf_mottle);
    } else if material > 1.5 {
        let bark_ridge = noise3(vec3f(local_pos.xy * 10.0, pos_world.z * 4.0) + seed.xyz * 17.0);
        albedo *= mix(0.62, 1.18, bark_ridge);
    } else {
        let soil_grain = noise3(pos_world * 8.0 + seed.yzx * 19.0);
        albedo *= mix(0.74, 1.12, soil_grain);
    }

    let wrap = diffuse * 0.7 + 0.3;
    let ambient = AMBIENT_COLOR * 0.74 + SKY_FILL * sky * 0.52;
    let specular = pow(max(dot(normal, half_dir), 0.0), 18.0) * select(0.025, 0.006, material > 2.5);
    let fresnel = pow(1.0 - saturate(dot(normal, view_dir)), 4.0) * select(0.03, 0.055, material > 2.5);

    return albedo * (ambient + wrap * 0.92) + vec3f(specular + fresnel);
}

@fragment
fn frag(in: FragmentInput) -> FragmentOutput {
    let seed = in.shape_params.w;
    let grain_seed = in.grain_params;
    let axis_scale = max(in.shape_params.xy, vec2f(0.001));
    let depth_scale = max(in.shape_params.z, 0.001);
    let radius_squared = dot(in.uv, in.uv);
    let local_seed = vec2f(grain_seed.x, grain_seed.y);
    let silhouette_noise = noise3(vec3f(in.uv * SILHOUETTE_NOISE_SCALE + local_seed * 11.0, seed * 31.0));
    let scallop_noise = noise3(vec3f(in.uv * 13.0 + local_seed.yx * 7.0, grain_seed.z * 19.0));
    let edge_radius = mix(0.78, 0.98, silhouette_noise) - 0.045 * scallop_noise;
    if radius_squared > edge_radius * edge_radius {
        discard;
    }

    let basis = cameraBillboardBasis(in.center_world);
    let z = sqrt(max(0.0, 1.0 - radius_squared));
    let ellipsoid_offset = basis[0] * in.uv.x * axis_scale.x
        + basis[1] * in.uv.y * axis_scale.y
        + basis[2] * z * depth_scale;
    let ellipsoid_normal = normalize(
        basis[0] * in.uv.x / axis_scale.x
        + basis[1] * in.uv.y / axis_scale.y
        + basis[2] * z / depth_scale
    );
    let local_pos = vec3f(in.uv, z);
    let facet_noise = hash33(floor(local_pos * 18.0 + grain_seed.xyz * 47.0));
    let facet_normal = normalize(ellipsoid_normal + (facet_noise - 0.5) * 0.23);
    let surface_pos_world = in.center_world + ellipsoid_offset * in.radius;
    let clip_pos = uniforms.viewProjMat * vec4f(surface_pos_world, 1);
    let depth = clip_pos.z / clip_pos.w;

    if clip_pos.w <= 0.0 || depth < 0.0 || depth > 1.0 {
        discard;
    }

    let plastic_volume = clamp(in.compression_volume_fac, 0.25, 1.5);
    let elastic_volume = clamp(in.elastic_volume_fac, 0.25, 1.5);
    let loose_factor = saturate((plastic_volume + elastic_volume) * 0.5);
    var color = shadeSnow(surface_pos_world, local_pos, facet_normal, loose_factor, grain_seed);
    if in.appearance.w > 0.5 {
        color = shadeForestParticle(surface_pos_world, local_pos, facet_normal, in.appearance, grain_seed);
    }

    var out: FragmentOutput;
    out.color = vec4f(color, 1.0);
    out.depth = depth;

    return out;
}
