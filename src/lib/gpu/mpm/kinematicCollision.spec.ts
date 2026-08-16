import { describe, expect, it } from "vitest";

import managerSrc from "./GpuMpmPipelineManager.ts?raw";
import fusedMlsSrc from "./fusedMlsG2p2g.cs.wgsl?raw";
import gridUpdateSrc from "./gridUpdate.cs.wgsl?raw";
import integrateParticlesSrc from "./integrateParticles.wgsl?raw";
import rigidPawColliderSrc from "../kinematicCollider/rigidPawSdfKinematicColliderPrelude.wgsl?raw";

const sourceBetween = (
    source: string,
    startMarker: string,
    endMarker: string,
) => {
    const start = source.indexOf(startMarker);
    const end = source.indexOf(endMarker, start);

    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);
    return source.slice(start, end);
};

describe("MPM rigid paw SDF collision integration", () => {
    it("binds rigid paw pose and field buffers for every collision pipeline", () => {
        expect(managerSrc).toContain(
            "kinematicColliderManager: GpuRigidPawSdfKinematicColliderManager",
        );
        expect(managerSrc).toContain(
            "{ binding: 11, resource: { buffer: kinematicColliderManager.poseBuffer } }",
        );
        expect(managerSrc).toContain(
            "{ binding: 12, resource: { buffer: kinematicColliderManager.sdfBuffer } }",
        );
        expect(managerSrc).toMatch(
            /KINEMATIC_COLLIDER_COUNT:\r?\n\s+kinematicColliderManager\.primitiveCount/,
        );
        expect(managerSrc).toMatch(
            /KINEMATIC_COLLIDER_SDF_RESOLUTION:\r?\n\s+kinematicColliderManager\.sdfResolution/,
        );
        expect(
            managerSrc.match(/\.\.\.kinematicColliderConstants/g),
        ).toHaveLength(4);
        expect(
            managerSrc.match(/rigidPawSdfColliderPreludeSrc}\\n/g),
        ).toHaveLength(3);
    });

    it("keeps collider storage out of bukkit-only pipeline layouts", () => {
        const bukkitSparseGridLayout = sourceBetween(
            managerSrc,
            "const bukkitSparseGridBindGroupLayout =",
            "const createBukkitSparseGridBindGroup",
        );
        const bukkitPipelineLayout = sourceBetween(
            managerSrc,
            "const bukkitBuildPipelineLayout =",
            "const finalizeNextActiveBlockDispatchPipelineLayout =",
        );
        const finalizePipelineLayout = sourceBetween(
            managerSrc,
            "const finalizeNextActiveBlockDispatchPipelineLayout =",
            "this.clearBukkitPipeline =",
        );

        expect(
            bukkitSparseGridLayout.match(/binding:/g),
        ).toHaveLength(1);
        expect(bukkitSparseGridLayout).toContain("binding: 0");
        expect(bukkitPipelineLayout).toContain(
            "bukkitSparseGridBindGroupLayout",
        );
        expect(finalizePipelineLayout).toContain(
            "bukkitSparseGridBindGroupLayout",
        );
        expect(bukkitPipelineLayout).not.toMatch(
            /^\s+sparseGridBindGroupLayout,$/m,
        );
        expect(finalizePipelineLayout).not.toMatch(
            /^\s+sparseGridBindGroupLayout,$/m,
        );
    });

    it("projects classic grid velocity in world space before its clamp", () => {
        const projection = gridUpdateSrc.lastIndexOf(
            "cell_velocity = projectVelocityAgainstKinematicCollider",
        );

        expect(gridUpdateSrc).toContain(
            "uniforms.gridMinCoords\n        + cellCenterGridCoord(block_cell_base, cell_offset)\n        * uniforms.gridCellDims",
        );
        expect(projection).toBeGreaterThan(
            gridUpdateSrc.lastIndexOf(
                "cell_velocity = applyDomainBoundaryVelocity",
            ),
        );
        expect(projection).toBeLessThan(
            gridUpdateSrc.lastIndexOf(
                "cell_velocity = clampVelocityToFixedPointRange",
            ),
        );
    });

    it("projects classic particle velocity before advection and recovers position after", () => {
        const velocityProjection = integrateParticlesSrc.indexOf(
            "particle.vel = projectVelocityAgainstKinematicCollider",
        );
        const advection = integrateParticlesSrc.indexOf(
            "particle.pos += particle.pos_displacement",
        );
        const recovery = integrateParticlesSrc.indexOf(
            "resolveParticlePositionAgainstKinematicCollider",
        );
        const staticCollision = integrateParticlesSrc.indexOf(
            "resolveParticleCollision",
        );

        expect(velocityProjection).toBeGreaterThanOrEqual(0);
        expect(velocityProjection).toBeLessThan(advection);
        expect(advection).toBeGreaterThanOrEqual(0);
        expect(advection).toBeLessThan(recovery);
        expect(recovery).toBeLessThan(staticCollision);
        expect(integrateParticlesSrc).toContain(
            "particle.pos - pre_advection_position",
        );
    });

    it("projects fused grid and particle velocities at Disney's stages", () => {
        const gridUpdate = sourceBetween(
            fusedMlsSrc,
            "fn loadTileVelocity",
            "fn velocityFromTile",
        );
        const particleIntegration = sourceBetween(
            fusedMlsSrc,
            "fn integrateFusedParticle",
            "fn scatterMlsParticleToGrid",
        );

        const gridVelocityProjection = gridUpdate.indexOf(
            "projectVelocityAgainstKinematicCollider",
        );
        const gridVelocityClamp = gridUpdate.indexOf(
            "clampVelocityToFixedPointRange(cell_velocity)",
        );
        expect(gridVelocityProjection).toBeGreaterThanOrEqual(0);
        expect(gridVelocityProjection).toBeLessThan(gridVelocityClamp);
        const velocityProjection = particleIntegration.indexOf(
            "(*particle).vel = projectVelocityAgainstKinematicCollider",
        );
        const advection = particleIntegration.indexOf(
            "(*particle).pos += (*particle).pos_displacement",
        );
        const finalContactSample = particleIntegration.indexOf(
            "let final_contact_sample = sampleKinematicColliderUnion",
        );
        const positionRecovery = particleIntegration.indexOf(
            "(*particle).pos -= final_contact_sample.signedDistance",
        );
        const finalVelocityProjection = particleIntegration.indexOf(
            "projectVelocityAgainstKinematicColliderSample(\n        final_contact_sample",
        );
        const correctedDisplacement = particleIntegration.indexOf(
            "(*particle).pos - pre_advection_position",
        );

        expect(velocityProjection).toBeGreaterThanOrEqual(0);
        expect(velocityProjection).toBeLessThan(advection);
        expect(advection).toBeGreaterThanOrEqual(0);
        expect(advection).toBeLessThan(finalContactSample);
        expect(finalContactSample).toBeLessThan(positionRecovery);
        expect(positionRecovery).toBeLessThan(finalVelocityProjection);
        expect(finalVelocityProjection).toBeLessThan(correctedDisplacement);
        expect(correctedDisplacement).toBeLessThan(
            particleIntegration.indexOf("resolveParticleCollision"),
        );
        expect(particleIntegration).toContain("resolveParticleCollision");
    });

    it("removes inward relative velocity from the recovered fused paw contact", () => {
        const contactProjection = sourceBetween(
            rigidPawColliderSrc,
            "fn projectVelocityAgainstKinematicColliderSample",
            "fn projectVelocityAgainstKinematicCollider(",
        );

        expect(contactProjection).toContain(
            "let relativeVelocity = velocity - sample.surfaceVelocity",
        );
        expect(contactProjection).toContain(
            "let normalVelocity = dot(relativeVelocity, sample.normal)",
        );
        expect(contactProjection).toContain(
            "relativeVelocity - normalVelocity * sample.normal",
        );
        expect(contactProjection).toMatch(
            /return sample\.surfaceVelocity\r?\n\s+\+ tangentialScale \* tangentialVelocity/,
        );

        const contactNormal = [0, 0, 1];
        const inwardRelativeVelocity = [1.25, 1.25, -3];
        const inwardNormalVelocity = inwardRelativeVelocity.reduce(
            (sum, component, index) =>
                sum + component * contactNormal[index],
            0,
        );
        const tangentialVelocity = inwardRelativeVelocity.map(
            (component, index) =>
                component - inwardNormalVelocity * contactNormal[index],
        );
        const projectedRelativeVelocity = tangentialVelocity.map(
            (component) => 0.8 * component,
        );
        const projectedRelativeNormalVelocity = projectedRelativeVelocity.reduce(
            (sum, component, index) =>
                sum + component * contactNormal[index],
            0,
        );

        expect(inwardNormalVelocity).toBeLessThan(0);
        expect(projectedRelativeNormalVelocity).toBeCloseTo(0);
    });
});
