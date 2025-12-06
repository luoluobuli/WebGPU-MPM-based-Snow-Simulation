// Marching cubes screen-space shading
// Soft lighting with SSS-style scattering
// Note: uniforms are at @group(0) @binding(0) via prelude

@group(0) @binding(1) var depthTexture: texture_depth_2d;
@group(0) @binding(2) var normalTexture: texture_2d<f32>;
@group(0) @binding(3) var shadedOutput: texture_storage_2d<rgba8unorm, write>;

const LIGHT_DIR = vec3f(0.5, 0.3, 0.8);
const AMBIENT_COLOR = vec3f(0.15, 0.18, 0.22);
const DIFFUSE_COLOR = vec3f(0.92, 0.94, 0.96);
const SPECULAR_COLOR = vec3f(0.3, 0.32, 0.35);
const DIFFUSE_STRENGTH = 0.6;
const SPECULAR_STRENGTH = 0.15;
const SHININESS = 8.0;

const SSS_STRENGTH = vec3f(0.1, 0.2, 0.25);
const SSS_COLOR = vec3f(0.75, 0.75, 0.75);

fn reconstructWorldPos(coords: vec2i, depth: f32, screenSize: vec2f) -> vec3f {
    let uv = (vec2f(coords) + 0.5) / screenSize;
    let ndc = vec2f(uv.x, 1.0 - uv.y) * 2.0 - 1.0;
    let clipPos = vec4f(ndc, depth, 1.0);
    let worldPosHom = uniforms.viewProjInvMat * clipPos;
    return worldPosHom.xyz / worldPosHom.w;
}

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let screenSize = vec2f(textureDimensions(depthTexture));
    let coords = vec2i(global_id.xy);
    
    if any(coords >= vec2i(screenSize)) {
        return;
    }
    
    let depth = textureLoad(depthTexture, coords, 0);
    
    if depth >= 1.0 {
        textureStore(shadedOutput, coords, vec4f(0.0, 0.0, 0.0, 0.0));
        return;
    }
    
    let normalData = textureLoad(normalTexture, coords, 0);
    var normal = normalData.xyz;
    let normalLen = length(normal);
    
    // Avoid NaNs from zero-length normals
    if (normalLen < 0.0001) {
        normal = vec3f(0.0, 1.0, 0.0); // Fallback to up vector
    } else {
        normal = normal / normalLen;
    }
    
    let worldPos = reconstructWorldPos(coords, depth, screenSize);
    let lightDir = normalize(vec3f(0.2, 0.8, 0.5)); // More top-down light
    let viewDir = normalize(uniforms.cameraPos - worldPos);
    
    // Diffuse lighting with wrap for soft snow look
    let NdotL = dot(normal, lightDir);
    let wrapDiffuse = (NdotL + 0.5) / 1.5; // Wrap lighting
    let diffuse = max(wrapDiffuse, 0.0);
    
    // Soft specular
    let halfDir = normalize(lightDir + viewDir);
    let specular = pow(max(dot(normal, halfDir), 0.0), SHININESS);
    
    // Subsurface scattering approximation
    // View-dependent transmission
    let VdotL = max(dot(-viewDir, lightDir), 0.0);
    let sss = pow(VdotL * SSS_STRENGTH, vec3f(1.5));
    
    // Fresnel-like rim lighting
    let fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);
    let rim = fresnel * 0.15;
    
    // Combine lighting
    var color = vec3f(0.35, 0.38, 0.42); // Higher ambient for visibility
    color += DIFFUSE_COLOR * diffuse * DIFFUSE_STRENGTH;
    color += SPECULAR_COLOR * specular * SPECULAR_STRENGTH;
    color += SSS_COLOR * sss;
    color += vec3f(rim);
    
    textureStore(shadedOutput, coords, vec4f(color, 1.0));
}
