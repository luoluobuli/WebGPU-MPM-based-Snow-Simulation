
@group(1) @binding(9) var<storage, read> colliderVertices: array<f32>;
@group(1) @binding(10) var<storage, read> colliderNormals: array<f32>;
@group(1) @binding(11) var<storage, read> colliderIndices: array<u32>;

fn getColliderVertex(index: u32) -> vec3f {
    let i = index * 3u;
    return vec3f(colliderVertices[i], colliderVertices[i + 1u], colliderVertices[i + 2u]);
}

fn closestPointTriangle(p: vec3f, a: vec3f, b: vec3f, c: vec3f) -> vec3f {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return a; }
    
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return b; }
    
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return a + v * ab;
    }
    
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return c; }
    
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return a + w * ac;
    }
    
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }
    
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return a + v * ab + w * ac;
}

fn intersectRayTriangle(origin: vec3f, dir: vec3f, v0: vec3f, v1: vec3f, v2: vec3f) -> f32 {
    let EPSILON = 1e-6;
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(dir, edge2);
    let a = dot(edge1, h);
    
    if (a > -EPSILON && a < EPSILON) { return -1.0; } // Parallel
    
    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) { return -1.0; }
    
    let q = cross(s, edge1);
    let v = f * dot(dir, q);
    if (v < 0.0 || u + v > 1.0) { return -1.0; }
    
    let t = f * dot(edge2, q);
    if (t > EPSILON) { return t; }
    
    return -1.0;
}

fn resolveParticleCollision(particle: ptr<function, ParticleData>) {
    // Broadphase: Check bounding box
    let minB = (uniforms.colliderTransformMat * vec4f(uniforms.colliderMinCoords, 1.0)).xyz; 
    let maxB = (uniforms.colliderTransformMat * vec4f(uniforms.colliderMaxCoords, 1.0)).xyz;
    
    let margin = 0.5;
    let safetyMin = min(minB, maxB) - vec3f(margin);
    let safetyMax = max(minB, maxB) + vec3f(margin);
    
    let currentPos = (*particle).pos;
    // For CCD, we need the "previous" position.
    // Assuming pos_displacement holds the step's movement vector.
    // Note: In GridToParticle, pos_displacement is (vel * dt).
    // In IntegrateParticles, pos_displacement is the accumulated displacement.
    let prevPos = currentPos - (*particle).pos_displacement;
    
    // Broadphase check on both positions (segment AABB)
    let minP = min(currentPos, prevPos);
    let maxP = max(currentPos, prevPos);
    
    if (any(minP > safetyMax) || any(maxP < safetyMin)) {
        return;
    }

    // Narrowphase Loop
    
    // Static Collision State
    var minInfoDistSq = 1e20;
    var closestNormal = vec3f(0.0, 0.0, 1.0);
    var closestPos = currentPos;
    
    // CCD State
    var minT = 1.0; // Max t is 1.0 (at currentPos)
    var hitNormal = vec3f(0.0, 0.0, 1.0);
    var hitPos = currentPos;
    var hasHit = false;

    let rayDir = (*particle).pos_displacement; // displacement vector
    // Only raycast if moved significantly
    let rayLength = length(rayDir);
    let doCCD = rayLength > 1e-4;

    let numIndices = arrayLength(&colliderIndices);
    if (numIndices == 0u) { return; }

    let transform = uniforms.colliderTransformMat;
    
    for (var i = 0u; i < numIndices; i += 3u) {
        let idx0 = colliderIndices[i];
        let idx1 = colliderIndices[i + 1u];
        let idx2 = colliderIndices[i + 2u];
        
        let v0_local = getColliderVertex(idx0);
        let v1_local = getColliderVertex(idx1);
        let v2_local = getColliderVertex(idx2);
        
        let v0 = (transform * vec4f(v0_local, 1.0)).xyz;
        let v1 = (transform * vec4f(v1_local, 1.0)).xyz;
        let v2 = (transform * vec4f(v2_local, 1.0)).xyz;
        
        // --- Static Check ---
        let cPoint = closestPointTriangle(currentPos, v0, v1, v2);
        let diff = currentPos - cPoint;
        let distSq = dot(diff, diff);
        
        if (distSq < minInfoDistSq) {
            minInfoDistSq = distSq;
            closestPos = cPoint;
            
            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let n = cross(edge1, edge2);
            let len = length(n);
            if (len > 1e-6) {
                closestNormal = n / len;
            }
        }
        
        // --- CCD Check ---
        if (doCCD) {
            let t = intersectRayTriangle(prevPos, rayDir, v0, v1, v2);
            if (t > 0.0 && t < minT) {
                 // Double-sided check
                 minT = t;
                 hasHit = true;
                 
                 // Compute intersection point exactly
                 hitPos = prevPos + rayDir * t;
                 
                 // Effective normal opposes ray direction
                 let faceN = normalize(cross(v1 - v0, v2 - v0));
                 if (dot(rayDir, faceN) < 0.0) {
                     hitNormal = faceN;
                 } else {
                     hitNormal = -faceN;
                 }
            }
        }
    }
    
    // Resolution Priority: CCD > Static
    
    // 1. CCD Response
    if (hasHit) {
        let surfaceMargin = 0.02;
        let snapPos = hitPos + hitNormal * surfaceMargin;
        let snapVec = snapPos - 
            (prevPos + (*particle).pos_displacement); // Vector from "intended" pos to "snapped" pos? 
            // Actually, we want snap from "current projected pos" to "surface".
            
        // Let's look at the correction vector:
        // particle.pos (current) -> snapPos
        // Correction = snapPos - particle.pos
        
        // Update Position
        (*particle).pos = snapPos;
        
        let oldVel = (*particle).pos_displacement / uniforms.simulationTimestep;
        var v_rel = oldVel - uniforms.colliderVelocity;
        let vn = dot(v_rel, hitNormal);
        
        // Velocity Response
        if (vn < 0.0) {
            let vN = vn * hitNormal;
            let vT = v_rel - vN;
            let friction = 0.1;
            
            // Standard bounce/friction response
            var newVel = vT * (1.0 - friction) + uniforms.colliderVelocity;
            
            // EXPERIMENTAL: If the snap distance was huge, it means we were crushed/tunneling deep.
            // The "oldVel" might be irrelevant or we shouldn't add "bounce" energy from the containment.
            // But we are reconstructing newVel based on oldVel reflected. That is physically okay-ish.
            
            // The EXPLOSION happens if we do:
            // vel = (snapPos - prevPos) / dt
            // Because (snapPos - prevPos) includes the "teleport" out of the wall.
            
            // My previous code did:
            // (*particle).pos_displacement = (*particle).pos - prevPos;
            // var v_rel = (*particle).pos_displacement / dt ...
            // This meant v_rel INCLUDED the teleport! That's the bug.
            
            // Fix: Calculate v_rel based on the particle's ARRIVAL velocity (before snap), not the SNAP velocity.
            // "oldVel" above is correct (displacement / dt). 
            // But wait, "pos_displacement" IS (pos - prevPos).
            // So if we update pos first, and then calc vel from pos_displacement, we carry the snap energy.
            
            // Correct flow:
            // 1. Calculate response velocity based on INCOMING velocity (oldVel).
            // 2. Set particle.vel = response velocity.
            // 3. Update particle.pos_displacement = particle.vel * dt (for consistency).
            // 4. Update particle.pos = snapPos (The snap is purely kinematic position correction, not dynamic).
            
            (*particle).vel = newVel;
            (*particle).pos_displacement = newVel * uniforms.simulationTimestep; 
        } else {
             // Moving away? Just update displacement to match new pos to prevent drift?
             // Or keep old velocity.
             (*particle).pos_displacement = (*particle).vel * uniforms.simulationTimestep; 
        }
        return; 
    }
    
    // 2. Static Response (Fallback)
    // Normal-Agnostic Push
    let dist = sqrt(minInfoDistSq);
    let diff = currentPos - closestPos;
    var pushDir = closestNormal;
    let lenDiff = length(diff);
    if (lenDiff > 1e-6) {
        pushDir = diff / lenDiff;
    }
    
    let threshold = 0.05; 

    if (dist < threshold) {
        let oldVel = (*particle).pos_displacement / uniforms.simulationTimestep;
        var v_rel = oldVel - uniforms.colliderVelocity;
        let vn = dot(v_rel, pushDir);

        if (vn < 0.0) {
            let vN = vn * pushDir;
            let vT = v_rel - vN;
            let friction = 0.0; 
            
            let newVel = vT * (1.0 - friction) + uniforms.colliderVelocity;
            
            (*particle).vel = newVel;
            (*particle).pos_displacement = newVel * uniforms.simulationTimestep;
        }
        
        // Position Correction: Enforce minimum distance
        let surfaceMargin = 0.02;
        if (dist < surfaceMargin) {
             (*particle).pos = closestPos + pushDir * surfaceMargin;
             // DO NOT Recalculate velocity from this new position.
             // We let the position "teleport" but the velocity "reflect" normally.
        }
    }
}
