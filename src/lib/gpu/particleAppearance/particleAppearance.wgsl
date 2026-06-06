fn decodeParticleAppearance(packed: u32) -> vec4f {
    let material = packed >> 24u;
    if material == 0u {
        return vec4f();
    }

    let r = f32(packed & 0xffu) / 255.0;
    let g = f32((packed >> 8u) & 0xffu) / 255.0;
    let b = f32((packed >> 16u) & 0xffu) / 255.0;

    return vec4f(r, g, b, f32(material));
}

fn particleAppearanceColor(appearance: vec4f, fallback: vec3f) -> vec3f {
    if appearance.w > 0.5 {
        return appearance.rgb;
    }

    return fallback;
}
