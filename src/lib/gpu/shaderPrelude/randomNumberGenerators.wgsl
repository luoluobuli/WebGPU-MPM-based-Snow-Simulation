// https://github.com/Cyan4973/xxHash
// https://gist.github.com/munrocket/236ed5ba7e409b8bdf1ff6eca5dcdc39
fn hash1(n: u32) -> u32 {
    var h32 = n + 374761393u;
    h32 = 668265263u * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = 2246822519u * (h32 ^ (h32 >> 15));
    h32 = 3266489917u * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

fn hash2(p: vec2u) -> u32 {
    let p2 = 2246822519u;
    let p3 = 3266489917u;
    let p4 = 668265263u;
    let p5 = 374761393u;

    var h32 = p.y + p5 + p.x * p3;
    h32 = p4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = p2 * (h32 ^ (h32 >> 15));
    h32 = p3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

fn hash3(p: vec3u) -> u32 {
    let p2 = 2246822519u;
    let p3 = 3266489917u;
    let p4 = 668265263u;
    let p5 = 374761393u;

    var h32 = p.z + p5 + p.x*p3;
    h32 = p4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * p3;
    h32 = p4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = p2 * (h32 ^ (h32 >> 15));
    h32 = p3 * (h32 ^ (h32 >> 13));
    
    return h32 ^ (h32 >> 16);
}

fn hash4(p: vec4u) -> u32 {
    let p2 = 2246822519u;
    let p3 = 3266489917u;
    let p4 = 668265263u;
    let p5 = 374761393u;

    var h32 = p.w + p5 + p.x * p3;
    h32 = p4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * p3;
    h32 = p4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.z  * p3;
    h32 = p4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = p2 * (h32 ^ (h32 >> 15));
    h32 = p3 * (h32 ^ (h32 >> 13));

    return h32 ^ (h32 >> 16);
}

fn noise3(x: vec3f) -> f32 {
    let p = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);

    let p_i = vec3i(p);
    let p_u = vec3u(bitcast<u32>(p_i.x), bitcast<u32>(p_i.y), bitcast<u32>(p_i.z));
    
    let n = p_u;
    
    let a = f32(hash3(n + vec3u(0,0,0))) / f32(0xFFFFFFFF);
    let b = f32(hash3(n + vec3u(1,0,0))) / f32(0xFFFFFFFF);
    let c = f32(hash3(n + vec3u(0,1,0))) / f32(0xFFFFFFFF);
    let d = f32(hash3(n + vec3u(1,1,0))) / f32(0xFFFFFFFF);
    let e = f32(hash3(n + vec3u(0,0,1))) / f32(0xFFFFFFFF);
    let f_val = f32(hash3(n + vec3u(1,0,1))) / f32(0xFFFFFFFF);
    let g = f32(hash3(n + vec3u(0,1,1))) / f32(0xFFFFFFFF);
    let h = f32(hash3(n + vec3u(1,1,1))) / f32(0xFFFFFFFF);
    
    return mix(
        mix(mix(a, b, u.x), mix(c, d, u.x), u.y),
        mix(mix(e, f_val, u.x), mix(g, h, u.x), u.y),
        u.z
    );
}

fn fbm(x: vec3f) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var p = x;
    for (var i = 0; i < 6; i++) {
        v += a * noise3(p);
        p = p * 2.0;
        a *= 0.5;
    }
    return v;
}

fn hash31(p: vec3f) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash33(p: vec3f) -> vec3f {
    var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}


fn gradientNoise(p: vec3f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    
    return mix(
        mix(
            mix(hash31(i + vec3f(0,0,0)), hash31(i + vec3f(1,0,0)), u.x),
            mix(hash31(i + vec3f(0,1,0)), hash31(i + vec3f(1,1,0)), u.x),
            u.y
        ),
        mix(
            mix(hash31(i + vec3f(0,0,1)), hash31(i + vec3f(1,0,1)), u.x),
            mix(hash31(i + vec3f(0,1,1)), hash31(i + vec3f(1,1,1)), u.x),
            u.y
        ),
        u.z
    ) * 2 - 1;
}

fn fbmNoise(p: vec3f, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;
    
    for (var i = 0; i < octaves; i++) {
        value += amplitude * gradientNoise(pos * frequency);
        frequency *= 2;
        amplitude *= 0.5;
    }
    return value;
}