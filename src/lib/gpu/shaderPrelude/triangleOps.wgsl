fn rayIntersectsTriangle(
    rayOrigin: vec3f,
    rayDir: vec3f,
    vert0: vec3f,
    vert1: vec3f,
    vert2: vec3f,
) -> bool {
    // Möller-Trumbore

    let EPSILON = 1e-6;
    
    let edge1 = vert1 - vert0;
    let edge2 = vert2 - vert0;
    
    let rayDirCrossEdge2 = cross(rayDir, edge2);
    let det = dot(edge1, rayDirCrossEdge2);
    
    // ray nearly parallel to triangle
    if abs(det) < EPSILON { return false; }
    
    let detInv = 1 / det;
    let originToVert0 = rayOrigin - vert0;


    let bary0 = dot(originToVert0, rayDirCrossEdge2) * detInv;
    if bary0 < 0 || 1 < bary0 { return false; }
    
    let originToVert0CrossEdge1 = cross(originToVert0, edge1);
    let bary1 = dot(rayDir, originToVert0CrossEdge1) * detInv;
    
    if bary1 < 0 || 1 < bary1 || bary0 + bary1 > 1 { return false; }
    
    let intersectionDist = dot(edge2, originToVert0CrossEdge1) * detInv;
    return intersectionDist > EPSILON;
}
