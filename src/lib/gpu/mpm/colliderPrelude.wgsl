fn getColliderVertex(index: u32) -> vec3f {
    let base_offset = uniforms.colliderNumIndices + index * 3u;
    let x = bitcast<f32>(colliderData[base_offset]);
    let y = bitcast<f32>(colliderData[base_offset + 1u]);
    let z = bitcast<f32>(colliderData[base_offset + 2u]);
    return vec3f(x, y, z);
}

fn closestPointTriangle(point: vec3f, tri_vert_a: vec3f, tri_vert_b: vec3f, tri_vert_c: vec3f) -> vec3f {
    let ab = tri_vert_b - tri_vert_a;
    let ac = tri_vert_c - tri_vert_a;
    let ap = point - tri_vert_a;
    
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return tri_vert_a; }
    
    let bp = point - tri_vert_b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return tri_vert_b; }
    
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return tri_vert_a + v * ab;
    }
    
    let cp = point - tri_vert_c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return tri_vert_c; }
    
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return tri_vert_a + w * ac;
    }
    
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return tri_vert_b + w * (tri_vert_c - tri_vert_b);
    }
    
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return tri_vert_a + v * ab + w * ac;
}

fn intersectRayTriangle(origin: vec3f, dir: vec3f, tri_vert_a: vec3f, tri_vert_b: vec3f, tri_vert_c: vec3f) -> f32 {
    let EPSILON = 1e-6;
    let edge1 = tri_vert_b - tri_vert_a;
    let edge2 = tri_vert_c - tri_vert_a;
    let h = cross(dir, edge2);
    let a = dot(edge1, h);
    
    if a > -EPSILON && a < EPSILON { return -1.0; } // Parallel
    
    let f = 1.0 / a;
    let s = origin - tri_vert_a;
    let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return -1.0; }
    
    let q = cross(s, edge1);
    let v = f * dot(dir, q);
    if v < 0.0 || u + v > 1.0 { return -1.0; }
    
    let t = f * dot(edge2, q);
    if t > EPSILON { return t; }
    
    return -1.0;
}

fn resolveParticleCollision(particle: ptr<function, ParticleData>) {
    // global broadphase (optional fast reject)
    let min_b = (uniforms.colliderTransformMat * vec4f(uniforms.colliderMinCoords, 1.0)).xyz; 
    let max_b = (uniforms.colliderTransformMat * vec4f(uniforms.colliderMaxCoords, 1.0)).xyz;
    
    let margin = 1.;
    let safety_min = min(min_b, max_b) - vec3f(margin);
    let safety_max = max(min_b, max_b) + vec3f(margin);
    
    let current_pos = (*particle).pos;
    let prev_pos = current_pos - (*particle).pos_displacement;
    
    let min_p = min(current_pos, prev_pos);
    let max_p = max(current_pos, prev_pos);
    
    // if strictly outside global bounds, return
    if any(min_p > safety_max) || any(max_p < safety_min) { return; }

    // setup for collision check
    var min_info_dist_sq = 1e20;
    var closest_normal = vec3f(0.0, 0.0, 1.0);
    var closest_pos = current_pos;
    
    var min_t = 1.; 
    var hit_normal = vec3f(0.0, 0.0, 1.0);
    var hit_pos = current_pos;
    var has_hit = false;

    let ray_dir = (*particle).pos_displacement; 
    let ray_length = length(ray_dir);
    let do_ccd = ray_length > 1e-4;

    let num_objects = uniforms.colliderNumObjects;
    if num_objects == 0u { return; }

    let transform = uniforms.colliderTransformMat;
    // Compute inverse transform for AABB check (world -> local)
    // Note: this assumes transform is invertible (which it should be)
    let invTransform = uniforms.colliderTransformInv;

    let current_pos_local = (invTransform * vec4f(current_pos, 1.0)).xyz;
    let prev_pos_local = (invTransform * vec4f(prev_pos, 1.0)).xyz;
    let min_p_local = min(current_pos_local, prev_pos_local);
    let max_p_local = max(current_pos_local, prev_pos_local);

    let local_margin = vec3f(margin); // Simplification: assuming uniform scale involved roughly matches 1.0 or we are conservative

    for (var i = 0u; i < num_objects; i++) {
        let obj = uniforms.objects[i];

        // AABB check in local space
        let safety_min_local = obj.min - local_margin;
        let safety_max_local = obj.max + local_margin;
        
        if any(min_p_local > safety_max_local) || any(max_p_local < safety_min_local) { 
            continue; 
        }

        // Narrowphase: check triangles
        let start = obj.startIndex;
        let end = start + obj.countIndices;

        for (var k = start; k < end; k += 3u) {
            let idx0 = colliderData[k];
            let idx1 = colliderData[k + 1u];
            let idx2 = colliderData[k + 2u];
            
            let v0_local = getColliderVertex(idx0);
            let v1_local = getColliderVertex(idx1);
            let v2_local = getColliderVertex(idx2);
            
            // Transform to world space for actual collision math
            let v0 = (transform * vec4f(v0_local, 1.0)).xyz;
            let v1 = (transform * vec4f(v1_local, 1.0)).xyz;
            let v2 = (transform * vec4f(v2_local, 1.0)).xyz;
            
            // static check
            let c_point = closestPointTriangle(current_pos, v0, v1, v2);
            let diff = current_pos - c_point;
            let dist_sq = dot(diff, diff);
            
            if dist_sq < min_info_dist_sq {
                min_info_dist_sq = dist_sq;
                closest_pos = c_point;
                
                let edge1 = v1 - v0;
                let edge2 = v2 - v0;
                let n = cross(edge1, edge2);
                let len = length(n);
                if len > 1e-6 {
                    closest_normal = n / len;
                }
            }
            
            // ccd check
            if do_ccd {
                let t = intersectRayTriangle(prev_pos, ray_dir, v0, v1, v2);
                if t > 0.0 && t < min_t {
                     min_t = t;
                     has_hit = true;
                     hit_pos = prev_pos + ray_dir * t;
                     let face_n = normalize(cross(v1 - v0, v2 - v0));
                     hit_normal = face_n * sign(dot(ray_dir, face_n)); // fix direction
                }
            }
        }
    }
    
    // Response logic (same as before)
    let velocity_scale_fac = 0.2 / uniforms.simulationTimestep;

    if has_hit {
        let surface_margin = 0.05;
        let snap_pos = hit_pos + hit_normal * surface_margin;
            
        (*particle).pos = snap_pos;
        
        let old_vel = (*particle).pos_displacement / uniforms.simulationTimestep;
        var v_rel = old_vel - uniforms.colliderVelocity;
        let vn = dot(v_rel, hit_normal);
        
        let v_n = vn * hit_normal;
        let v_t = v_rel - v_n;
        let friction = 0.1;
        
        var new_vel: vec3f;
        if vn < 0.0 {
            new_vel = v_t * (1.0 - friction) + uniforms.colliderVelocity * velocity_scale_fac;
        } else {
            new_vel = v_rel + uniforms.colliderVelocity * velocity_scale_fac;
        }
        
        (*particle).vel = new_vel;
        (*particle).pos_displacement = new_vel * uniforms.simulationTimestep; 
        return; 
    }
    
    let dist = sqrt(min_info_dist_sq);
    let diff = current_pos - closest_pos;
    var push_dir = closest_normal;
    let len_diff = length(diff);
    if len_diff > 1e-6 {
        push_dir = diff / len_diff;
    }
    
    let threshold = 0.05; 

    if dist < threshold {
        let old_vel = (*particle).pos_displacement / uniforms.simulationTimestep;
        let v_rel = old_vel - uniforms.colliderVelocity;
        let vn = dot(v_rel, push_dir);

        let v_n = vn * push_dir;
        let v_t = v_rel - v_n;
        let friction = 0.0;
        
        var new_vel: vec3f;
        if vn < 0.0 {
            new_vel = v_t * (1.0 - friction) + uniforms.colliderVelocity * velocity_scale_fac;
        } else {
            new_vel = v_rel + uniforms.colliderVelocity * velocity_scale_fac;
        }
        
        (*particle).vel = new_vel;
        (*particle).pos_displacement = new_vel * uniforms.simulationTimestep;
        
        let surface_margin = 0.05;
        if dist < surface_margin {
            (*particle).pos = closest_pos + push_dir * surface_margin;
        }
    }
}

fn pointInsideAABB(p: vec3f, minB: vec3f, maxB: vec3f) -> bool {
    return all(p >= minB) && all(p <= maxB);
}

fn closestPointAABB(p: vec3f, minB: vec3f, maxB: vec3f) -> vec3f {
    return clamp(p, minB, maxB);
}

fn aabbNormal(p: vec3f, minB: vec3f, maxB: vec3f) -> vec3f {
    let dMin = abs(p - minB);
    let dMax = abs(maxB - p);

    let minDist = min(min(dMin.x, dMin.y), min(dMin.z, min(dMax.x, min(dMax.y, dMax.z))));

    if (dMin.x == minDist) { return vec3f(-1, 0, 0); }
    if (dMax.x == minDist) { return vec3f( 1, 0, 0); }
    if (dMin.y == minDist) { return vec3f(0, -1, 0); }
    if (dMax.y == minDist) { return vec3f(0,  1, 0); }
    if (dMin.z == minDist) { return vec3f(0, 0, -1); }
    return vec3f(0, 0, 1);
}

fn rayAABB(ro: vec3f, rd: vec3f, minB: vec3f, maxB: vec3f) -> f32 {
    let invD = 1.0 / rd;

    let t0 = (minB - ro) * invD;
    let t1 = (maxB - ro) * invD;

    let tmin = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let tmax = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));

    if (tmax < 0.0 || tmin > tmax) {
        return -1.0;
    }

    return tmin;
}

fn resolveDynamicParticleCollision(particle: ptr<function, ParticleData>) {
    // global broadphase (optional fast reject)
    let min_b = (uniforms.colliderTransformMat * vec4f(uniforms.colliderMinCoords, 1.0)).xyz; 
    let max_b = (uniforms.colliderTransformMat * vec4f(uniforms.colliderMaxCoords, 1.0)).xyz;
    
    let margin = 1.;
    let safety_min = min(min_b, max_b) - vec3f(margin);
    let safety_max = max(min_b, max_b) + vec3f(margin);
    
    let current_pos = (*particle).pos;
    let prev_pos = current_pos - (*particle).pos_displacement;
    
    let min_p = min(current_pos, prev_pos);
    let max_p = max(current_pos, prev_pos);
    
    // if strictly outside global bounds, return
    if any(min_p > safety_max) || any(max_p < safety_min) { return; }

    // setup for collision check
    var min_info_dist_sq = 1e20;
    var closest_normal = vec3f(0.0, 0.0, 1.0);
    var closest_pos = current_pos;
    
    var min_t = 1.; 
    var hit_normal = vec3f(0.0, 0.0, 1.0);
    var hit_pos = current_pos;
    var has_hit = false;

    let ray_dir = (*particle).pos_displacement; 
    let ray_length = length(ray_dir);
    let do_ccd = ray_length > 1e-4;

    let num_objects = uniforms.colliderNumObjects;
    if num_objects == 0u { return; }

    let transform = uniforms.colliderTransformMat;
    // Compute inverse transform for AABB check (world -> local)
    // Note: this assumes transform is invertible (which it should be)
    let invTransform = uniforms.colliderTransformInv;

    let current_pos_local = (invTransform * vec4f(current_pos, 1.0)).xyz;
    let prev_pos_local = (invTransform * vec4f(prev_pos, 1.0)).xyz;
    let ray_dir_local = current_pos_local - prev_pos_local;
    let min_p_local = min(current_pos_local, prev_pos_local);
    let max_p_local = max(current_pos_local, prev_pos_local);

    let local_margin = vec3f(margin); // Simplification: assuming uniform scale involved roughly matches 1.0 or we are conservative
    for (var i = 0u; i < num_objects; i++) {
        let obj = uniforms.objects[i];

        if any(min_p_local > obj.max + local_margin) || any(max_p_local < obj.min - local_margin) {
            continue;
        }

        let c = closestPointAABB(current_pos_local, obj.min, obj.max);
        let diff = current_pos_local - c;
        let dist_sq = dot(diff, diff);

        if dist_sq < min_info_dist_sq {
            min_info_dist_sq = dist_sq;
            closest_pos = (transform * vec4f(c, 1.0)).xyz;
            closest_normal = (transform * vec4f(aabbNormal(c, obj.min, obj.max), 0.0)).xyz;
        }

        if do_ccd {
            let t = rayAABB(prev_pos_local, ray_dir_local, obj.min, obj.max);
            if t >= 0.0 && t < min_t {
                has_hit = true;
                min_t = t;
                hit_pos = prev_pos + ray_dir * t;
                hit_normal = closest_normal;
            }
        }
    }
    
    // Response logic (same as before)
    let velocity_scale_fac = 0.2 / uniforms.simulationTimestep;

    if has_hit {
        let surface_margin = 0.05;
        let snap_pos = hit_pos + hit_normal * surface_margin;
            
        (*particle).pos = snap_pos;
        
        let old_vel = (*particle).pos_displacement / uniforms.simulationTimestep;
        var v_rel = old_vel - uniforms.colliderVelocity;
        let vn = dot(v_rel, hit_normal);
        
        let v_n = vn * hit_normal;
        let v_t = v_rel - v_n;
        let friction = 0.1;
        
        var new_vel: vec3f;
        if vn < 0.0 {
            new_vel = v_t * (1.0 - friction) + uniforms.colliderVelocity * velocity_scale_fac;
        } else {
            new_vel = v_rel + uniforms.colliderVelocity * velocity_scale_fac;
        }
        
        (*particle).vel = new_vel;
        (*particle).pos_displacement = new_vel * uniforms.simulationTimestep; 
        return; 
    }
    
    let dist = sqrt(min_info_dist_sq);
    let diff = current_pos - closest_pos;
    var push_dir = closest_normal;
    let len_diff = length(diff);
    if len_diff > 1e-6 {
        push_dir = diff / len_diff;
    }
    
    let threshold = 0.05; 

    if dist < threshold {
        let old_vel = (*particle).pos_displacement / uniforms.simulationTimestep;
        let v_rel = old_vel - uniforms.colliderVelocity;
        let vn = dot(v_rel, push_dir);

        let v_n = vn * push_dir;
        let v_t = v_rel - v_n;
        let friction = 0.0;
        
        var new_vel: vec3f;
        if vn < 0.0 {
            new_vel = v_t * (1.0 - friction) + uniforms.colliderVelocity * velocity_scale_fac;
        } else {
            new_vel = v_rel + uniforms.colliderVelocity * velocity_scale_fac;
        }
        
        (*particle).vel = new_vel;
        (*particle).pos_displacement = new_vel * uniforms.simulationTimestep;
        
        let surface_margin = 0.05;
        if dist < surface_margin {
            (*particle).pos = closest_pos + push_dir * surface_margin;
        }
    }
}
