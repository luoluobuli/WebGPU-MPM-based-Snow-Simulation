@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct FragmentInput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) center_world: vec3f,
    @location(2) radius: f32,
    @location(3) compression_volume_fac: f32,
    @location(4) elastic_volume_fac: f32,
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

fn perturbSnowNormal(pos_world: vec3f, normal_world: vec3f, loose_factor: f32) -> vec3f {
    let grain_gradient = noiseGradient3(pos_world * NOISE_SCALE);
    let tangent_gradient = grain_gradient - dot(grain_gradient, normal_world) * normal_world;
    let noise_strength = mix(0.035, 0.11, loose_factor);

    return normalize(normal_world + tangent_gradient * noise_strength);
}

fn shadeSnow(pos_world: vec3f, normal_world: vec3f, loose_factor: f32) -> vec3f {
    let normal = perturbSnowNormal(pos_world, normal_world, loose_factor);
    let view_dir = normalize(uniforms.cameraPos - pos_world);
    let half_dir = normalize(LIGHT_DIR + view_dir);

    let diffuse = max(dot(normal, LIGHT_DIR), 0.0);
    let wrap_diffuse = diffuse * 0.74 + 0.26;
    let sky = max(normal.z * 0.5 + 0.5, 0.0);
    let fresnel = pow(1.0 - saturate(dot(normal, view_dir)), 3.0);

    let grain = noise3(pos_world * 38.0);
    let grain_tint = mix(0.9, 1.08, grain);
    let specular = pow(max(dot(normal, half_dir), 0.0), 18.0) * 0.16;
    let glint_cell = floor(pos_world * mix(80.0, 130.0, loose_factor));
    let glint_seed = hash31(glint_cell + vec3f(13.0, 37.0, 71.0));
    let glint = smoothstep(0.985, 1.0, glint_seed) * pow(max(dot(normal, half_dir), 0.0), 6.0);

    let diffuse_color = SNOW_ALBEDO * wrap_diffuse * grain_tint;
    let ambient = AMBIENT_COLOR + SKY_FILL * sky;

    return diffuse_color + ambient + vec3f(specular + glint * 0.45) + fresnel * vec3f(0.045, 0.06, 0.08);
}

@fragment
fn frag(in: FragmentInput) -> FragmentOutput {
    let radius_squared = dot(in.uv, in.uv);
    if radius_squared > 1.0 {
        discard;
    }

    let basis = cameraBillboardBasis(in.center_world);
    let z = sqrt(max(0.0, 1.0 - radius_squared));
    let normal_world = normalize(basis[0] * in.uv.x + basis[1] * in.uv.y + basis[2] * z);
    let surface_pos_world = in.center_world + normal_world * in.radius;
    let clip_pos = uniforms.viewProjMat * vec4f(surface_pos_world, 1);
    let depth = clip_pos.z / clip_pos.w;

    if clip_pos.w <= 0.0 || depth < 0.0 || depth > 1.0 {
        discard;
    }

    let plastic_volume = clamp(in.compression_volume_fac, 0.25, 1.5);
    let elastic_volume = clamp(in.elastic_volume_fac, 0.25, 1.5);
    let loose_factor = saturate((plastic_volume + elastic_volume) * 0.5);
    let color = shadeSnow(surface_pos_world, normal_world, loose_factor);

    var out: FragmentOutput;
    out.color = vec4f(color, 1.0);
    out.depth = depth;

    return out;
}
