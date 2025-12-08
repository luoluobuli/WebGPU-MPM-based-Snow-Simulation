// BVH Node structure (32 bytes)
// min: vec3f, leftChildOrPrimIndex: u32
// max: vec3f, rightChildOrPrimCount: u32 
// If high bit (0x80000000) is set: leaf node, lower bits = primitive count
// If high bit is clear: internal node, value = right child index
struct BvhNode {
    min: vec3f,
    leftChildOrPrimIndex: u32,
    max: vec3f,
    rightChildOrPrimCount: u32,
}

const BVH_LEAF_FLAG = 0x80000000u;

fn bvhNodeIsLeaf(node: BvhNode) -> bool {
    return (node.rightChildOrPrimCount & BVH_LEAF_FLAG) != 0u;
}

fn bvhNodePrimitiveCount(node: BvhNode) -> u32 {
    return node.rightChildOrPrimCount & 0x7FFFFFFFu;
}

fn bvhNodeRightChild(node: BvhNode) -> u32 {
    return node.rightChildOrPrimCount;
}

@group(1) @binding(10) var<storage, read> bvhNodes: array<BvhNode>;

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

// AABB intersection test for BVH traversal
fn intersectAABB(ray_origin: vec3f, ray_dir_inv: vec3f, box_min: vec3f, box_max: vec3f) -> bool {
    let t1 = (box_min - ray_origin) * ray_dir_inv;
    let t2 = (box_max - ray_origin) * ray_dir_inv;
    
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    
    let t_enter = max(max(tmin.x, tmin.y), tmin.z);
    let t_exit = min(min(tmax.x, tmax.y), tmax.z);
    
    return t_enter <= t_exit && t_exit >= 0.0;
}

// Point-AABB distance check for closest point queries
fn pointInOrNearAABB(point: vec3f, box_min: vec3f, box_max: vec3f, threshold_sq: f32) -> bool {
    // Compute squared distance from point to AABB
    var dist_sq = 0.0;
    
    if point.x < box_min.x {
        let d = box_min.x - point.x;
        dist_sq += d * d;
    } else if point.x > box_max.x {
        let d = point.x - box_max.x;
        dist_sq += d * d;
    }
    
    if point.y < box_min.y {
        let d = box_min.y - point.y;
        dist_sq += d * d;
    } else if point.y > box_max.y {
        let d = point.y - box_max.y;
        dist_sq += d * d;
    }
    
    if point.z < box_min.z {
        let d = box_min.z - point.z;
        dist_sq += d * d;
    } else if point.z > box_max.z {
        let d = point.z - box_max.z;
        dist_sq += d * d;
    }
    
    return dist_sq <= threshold_sq;
}


const BVH_STACK_SIZE = 32u;

fn resolveParticleCollision(particle: ptr<function, ParticleData>) {
    // Global broadphase (optional fast reject)
    let min_b = (uniforms.colliderTransformMat * vec4f(uniforms.colliderMinCoords, 1.0)).xyz; 
    let max_b = (uniforms.colliderTransformMat * vec4f(uniforms.colliderMaxCoords, 1.0)).xyz;
    
    let margin = 1.;
    let safety_min = min(min_b, max_b) - vec3f(margin);
    let safety_max = max(min_b, max_b) + vec3f(margin);
    
    let current_pos = (*particle).pos;
    let prev_pos = current_pos - (*particle).pos_displacement;
    
    let min_p = min(current_pos, prev_pos);
    let max_p = max(current_pos, prev_pos);
    
    // If strictly outside global bounds, return
    if any(min_p > safety_max) || any(max_p < safety_min) { return; }

    let num_objects = uniforms.colliderNumObjects;
    if num_objects == 0u { return; }

    // Setup for collision check
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

    let transform = uniforms.colliderTransformMat;
    let invTransform = uniforms.colliderTransformInv;

    // Transform positions to local space for BVH traversal
    let current_pos_local = (invTransform * vec4f(current_pos, 1.0)).xyz;
    let prev_pos_local = (invTransform * vec4f(prev_pos, 1.0)).xyz;
    let ray_dir_local = current_pos_local - prev_pos_local;

    // Compute inverse ray direction for AABB tests (in local space)
    var ray_dir_inv_local = vec3f(1e10, 1e10, 1e10);
    if abs(ray_dir_local.x) > 1e-8 { ray_dir_inv_local.x = 1.0 / ray_dir_local.x; }
    if abs(ray_dir_local.y) > 1e-8 { ray_dir_inv_local.y = 1.0 / ray_dir_local.y; }
    if abs(ray_dir_local.z) > 1e-8 { ray_dir_inv_local.z = 1.0 / ray_dir_local.z; }

    // BVH traversal using fixed-size stack
    var stack: array<u32, BVH_STACK_SIZE>;
    var stack_ptr = 0u;
    stack[stack_ptr] = 0u; // Start with root node
    stack_ptr += 1u;

    let local_margin = vec3f(margin);
    let threshold_sq = min_info_dist_sq;

    // Pre-compute local path bounds for BVH traversal
    let path_min_local = min(current_pos_local, prev_pos_local);
    let path_max_local = max(current_pos_local, prev_pos_local);

    while stack_ptr > 0u {
        stack_ptr -= 1u;
        let node_idx = stack[stack_ptr];
        let node = bvhNodes[node_idx];

        // Check if particle path intersects this node's AABB (in local space)
        // Note: node.min/max are already in local space
        let box_min = node.min - local_margin;
        let box_max = node.max + local_margin;
        
        if any(path_min_local > box_max) || any(path_max_local < box_min) {
            continue;
        }

        if bvhNodeIsLeaf(node) {
            // Leaf node - test triangles
            let start = node.leftChildOrPrimIndex;
            let prim_count = bvhNodePrimitiveCount(node);
            let end = start + prim_count;

            for (var i = start; i < end; i++) {
                let tri_base = i * 3u;
                let idx0 = colliderData[tri_base];
                let idx1 = colliderData[tri_base + 1u];
                let idx2 = colliderData[tri_base + 2u];
                
                let v0_local = getColliderVertex(idx0);
                let v1_local = getColliderVertex(idx1);
                let v2_local = getColliderVertex(idx2);
                
                // Transform to world space for collision math
                let v0 = (transform * vec4f(v0_local, 1.0)).xyz;
                let v1 = (transform * vec4f(v1_local, 1.0)).xyz;
                let v2 = (transform * vec4f(v2_local, 1.0)).xyz;
                
                // Static closest point check
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
                
                // CCD ray-triangle intersection
                if do_ccd {
                    let t = intersectRayTriangle(prev_pos, ray_dir, v0, v1, v2);
                    if t > 0.0 && t < min_t {
                        min_t = t;
                        has_hit = true;
                        hit_pos = prev_pos + ray_dir * t;
                        let face_n = normalize(cross(v1 - v0, v2 - v0));
                        hit_normal = face_n * sign(dot(ray_dir, face_n));
                    }
                }
            }
        } else {
            // Internal node - push children onto stack
            if stack_ptr < BVH_STACK_SIZE - 1u {
                let left_child = node.leftChildOrPrimIndex;
                let right_child = bvhNodeRightChild(node);
                stack[stack_ptr] = left_child;
                stack_ptr += 1u;
                stack[stack_ptr] = right_child;
                stack_ptr += 1u;
            }
        }
    }
    
    // Response logic
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
