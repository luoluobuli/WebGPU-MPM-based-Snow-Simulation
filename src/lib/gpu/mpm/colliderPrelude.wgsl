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
    // broadphase: check bounding box
    let min_b = (uniforms.colliderTransformMat * vec4f(uniforms.colliderMinCoords, 1.0)).xyz; 
    let max_b = (uniforms.colliderTransformMat * vec4f(uniforms.colliderMaxCoords, 1.0)).xyz;
    
    let margin = 0.5;
    let safety_min = min(min_b, max_b) - vec3f(margin);
    let safety_max = max(min_b, max_b) + vec3f(margin);
    
    let current_pos = (*particle).pos;
    let prev_pos = current_pos - (*particle).pos_displacement;
    
    // broadphase check on both positions (segment AABB)
    let min_p = min(current_pos, prev_pos);
    let max_p = max(current_pos, prev_pos);
    
    if any(min_p > safety_max) || any(max_p < safety_min) { return; }

    // narrowphase loop
    
    // static collision state
    var min_info_dist_sq = 1e20;
    var closest_normal = vec3f(0.0, 0.0, 1.0);
    var closest_pos = current_pos;
    
    // CCD state
    var min_t = 1.; // max t is 1.0 (at current_pos)
    var hit_normal = vec3f(0.0, 0.0, 1.0);
    var hit_pos = current_pos;
    var has_hit = false;

    let ray_dir = (*particle).pos_displacement; // displacement vector
    // only raycast if moved significantly
    let ray_length = length(ray_dir);
    let do_ccd = ray_length > 1e-4;

    let num_indices = uniforms.colliderNumIndices;
    if num_indices == 0u { return; }

    let transform = uniforms.colliderTransformMat;
    
    for (var i = 0u; i < num_indices; i += 3u) {
        let idx0 = colliderData[i];
        let idx1 = colliderData[i + 1u];
        let idx2 = colliderData[i + 2u];
        
        let v0_local = getColliderVertex(idx0);
        let v1_local = getColliderVertex(idx1);
        let v2_local = getColliderVertex(idx2);
        
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
                 // double-sided check
                 min_t = t;
                 has_hit = true;
                 
                 // compute intersection point exactly
                 hit_pos = prev_pos + ray_dir * t;
                 
                 // effective normal opposes ray direction
                 let face_n = normalize(cross(v1 - v0, v2 - v0));
                 hit_normal *= sign(dot(ray_dir, face_n));
            }
        }
    }

    let velocity_scale_fac = 0.05 / uniforms.simulationTimestep;
    
    // ccd resolved over static
    
    // ccd response
    if has_hit {
        let surface_margin = 0.02;
        let snap_pos = hit_pos + hit_normal * surface_margin;
            
        (*particle).pos = snap_pos;
        
        let old_vel = (*particle).pos_displacement / uniforms.simulationTimestep;
        var v_rel = old_vel - uniforms.colliderVelocity;
        let vn = dot(v_rel, hit_normal);
        
        // decompose relative velocity
        let v_n = vn * hit_normal;
        let v_t = v_rel - v_n;
        let friction = 0.1;
        
        // velocity response: remove normal component (if penetrating) and add collider velocity
        var new_vel: vec3f;
        if vn < 0.0 {
            // penetrating: remove inward normal velocity, apply friction to tangent
            new_vel = v_t * (1.0 - friction) + uniforms.colliderVelocity * velocity_scale_fac;
        } else {
            // moving away: keep relative velocity but still add collider velocity base
            new_vel = v_rel + uniforms.colliderVelocity * velocity_scale_fac;
        }
        
        (*particle).vel = new_vel;
        (*particle).pos_displacement = new_vel * uniforms.simulationTimestep; 
        return; 
    }
    
    // normal-agnostic push
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

        // decompose relative velocity
        let v_n = vn * push_dir;
        let v_t = v_rel - v_n;
        let friction = 0.0;
        
        // velocity response: remove normal component (if penetrating) and add collider velocity
        var new_vel: vec3f;
        if vn < 0.0 {
            // penetrating: remove inward normal velocity, apply friction to tangent
            new_vel = v_t * (1.0 - friction) + uniforms.colliderVelocity * velocity_scale_fac;
        } else {
            // moving away: keep relative velocity but still add collider velocity base
            new_vel = v_rel + uniforms.colliderVelocity * velocity_scale_fac;
        }
        
        (*particle).vel = new_vel;
        (*particle).pos_displacement = new_vel * uniforms.simulationTimestep;
        
        // position correction
        let surface_margin = 0.02;
        if dist < surface_margin {
            (*particle).pos = closest_pos + push_dir * surface_margin;
        }
    }
}
