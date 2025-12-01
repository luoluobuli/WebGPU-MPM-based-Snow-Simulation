struct Ray {
    origin: vec3f,
    dir: vec3f,
}

fn calculateViewRay(uv: vec2f, texture_dims: vec2u) -> Ray {
    let uvNormalized = vec2f(uv.x, 1 - uv.y) * 2 - 1;
    
    let nearPosHom = uniforms.viewProjInvMat * vec4f(uvNormalized, 0, 1);
    let nearPos = nearPosHom.xyz / nearPosHom.w;
    
    let farPosHom = uniforms.viewProjInvMat * vec4f(uvNormalized, 1, 1);
    let farPos = farPosHom.xyz / farPosHom.w;

    let ray_origin = nearPos;
    let ray_dir = normalize(farPos - nearPos);

    return Ray(ray_origin, ray_dir);
}