import type { Mat4 } from "wgpu-matrix";
import { GRAVITATIONAL_ACCELERATION_M_PER_S2 } from "$lib/gpu/gravity";

const FIXED_POINT_SCALE = 65536;
const INVERSE_PARTICLE_DENSITY = 1 / 400;

export class GpuUniformsBufferManager {
    private readonly device: GPUDevice;
    private readonly float1Upload = new Float32Array(1);
    private readonly uint1Upload = new Uint32Array(1);
    private readonly float3Upload = new Float32Array(3);
    private readonly uint3Upload = new Uint32Array(3);
    private readonly maxStableParticleUpload = new Float32Array(4);
    private readonly timestepDerivedUpload = new Float32Array(12);
    private simulationTimestepS = 0;
    private interactionStrength = 0;
    private invGridCellDims: [number, number, number] = [0, 0, 0];
    private invGridCellDimsSquared: [number, number, number] = [0, 0, 0];

    readonly buffer: GPUBuffer;
    readonly bindGroupLayout: GPUBindGroupLayout;
    readonly bindGroup: GPUBindGroup;

    constructor({
        device,
    }: {
        device: GPUDevice,
    }) {
        const uniformsBuffer = device.createBuffer({
            label: "uniforms buffer",
            size: 37376,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
        });

        const uniformsBindGroupLayout = device.createBindGroupLayout({
            label: "uniforms bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform",
                    },
                },
            ],
        });
        const uniformsBindGroup = device.createBindGroup({
            label: "uniforms bind group",
            layout: uniformsBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: uniformsBuffer,
                    },
                },
            ],
        });


        this.device = device;

        this.buffer = uniformsBuffer;
        this.bindGroupLayout = uniformsBindGroupLayout;
        this.bindGroup = uniformsBindGroup;
    }

    private writeFloat32(offset: number, value: number) {
        this.float1Upload[0] = value;
        this.device.queue.writeBuffer(this.buffer, offset, this.float1Upload);
    }

    private writeUint32(offset: number, value: number) {
        this.uint1Upload[0] = value;
        this.device.queue.writeBuffer(this.buffer, offset, this.uint1Upload);
    }

    private writeFloat32x3(offset: number, value: [number, number, number]) {
        this.float3Upload.set(value);
        this.device.queue.writeBuffer(this.buffer, offset, this.float3Upload);
    }

    private writeUint32x3(offset: number, value: [number, number, number]) {
        this.uint3Upload.set(value);
        this.device.queue.writeBuffer(this.buffer, offset, this.uint3Upload);
    }

    writeSimulationTimestepS(timestep: number) {
        const finiteTimestep = Number.isFinite(timestep) ? timestep : 0;
        this.simulationTimestepS = finiteTimestep;
        this.writeFloat32(0, timestep);
        this.writeFloat32(8, finiteTimestep > 0 ? 1 / finiteTimestep : 0);
        this.writeTimestepDerivedValues();
    }

    writeFixedPointScale(fixedPointScale: number) {
        this.writeFloat32(4, fixedPointScale);
    }

    writeTime(time: number) {
        this.writeUint32(12, time);
    }

    writeGridMinCoords(min: [number, number, number]) {
        this.writeFloat32x3(16, min);
    }

    writeGridMaxCoords(max: [number, number, number]) {
        this.writeFloat32x3(32, max);
    }

    writeViewProjMat(viewProjMat: Mat4) {
        this.device.queue.writeBuffer(this.buffer, 48, viewProjMat.buffer);
    }

    writeViewProjInvMat(viewProjInvMat: Mat4) {
        this.device.queue.writeBuffer(this.buffer, 112, viewProjInvMat.buffer);
    }

    writeMeshMinCoords(min: [number, number, number]) {
        this.writeFloat32x3(176, min);
    }

    writeMeshMaxCoords(max: [number, number, number]) {
        this.writeFloat32x3(192, max);
    }

    writeGridResolution(gridResolution: [number, number, number]) {
        this.writeUint32x3(208, gridResolution);
    }

    writeColliderMinCoords(min: [number, number, number]) {
        this.writeFloat32x3(224, min);
    }

    writeColliderMaxCoords(max: [number, number, number]) {
        this.writeFloat32x3(240, max);
    }

    writeColliderTransformMat(transformMat: Mat4) {
        this.device.queue.writeBuffer(this.buffer, 256, transformMat.buffer);
    }

    writeColliderVel(vel: [number, number, number]) {
        this.writeFloat32x3(320, vel);
    }

    writeCameraPos(cameraPos: [number, number, number]) {
        this.writeFloat32x3(336, cameraPos);
    }

    writeColliderTransformIsIdentity(isIdentity: boolean) {
        this.writeUint32(348, isIdentity ? 1 : 0);
    }

    writeGridCellDims(gridCellDims: [number, number, number]) {
        this.writeFloat32x3(352, gridCellDims);
    }

    writeColliderVelocityIsZero(isZero: boolean) {
        this.writeUint32(364, isZero ? 1 : 0);
    }

    writeInvGridCellDims(invGridCellDims: [number, number, number]) {
        this.invGridCellDims = invGridCellDims;
        this.writeFloat32x3(624, invGridCellDims);
        this.writeTimestepDerivedValues();
    }

    writeInvGridCellDimsSquared(invGridCellDimsSquared: [number, number, number]) {
        this.invGridCellDimsSquared = invGridCellDimsSquared;
        this.writeFloat32x3(640, invGridCellDimsSquared);
        this.writeTimestepDerivedValues();
    }

    writeSimulationDomainDerivedValues({
        center,
        maxInside,
    }: {
        center: [number, number, number],
        maxInside: [number, number, number],
    }) {
        this.writeFloat32x3(656, center);
        this.writeFloat32x3(672, maxInside);
    }

    writeLightViewProjMat(lightViewProjMat: Mat4) {
        this.device.queue.writeBuffer(this.buffer, 368, lightViewProjMat.buffer);
    }

    writeColliderTransformInv(transformInv: Mat4) {
        this.device.queue.writeBuffer(this.buffer, 432, transformInv.buffer);
    }

    // Interaction Uniforms (Start 496)
    writeInteractionPos(pos: [number, number, number]) {
        this.writeFloat32x3(496, pos);
    }

    writeInteractionStrength(strength: number) {
        this.interactionStrength = Number.isFinite(strength) ? strength : 0;
        this.writeFloat32(508, strength);
        this.writeInteractionStrengthDelta();
    }

    writeInteractionRadius(radius: number) {
        this.writeFloat32(512, radius);
        this.writeFloat32(540, radius * radius);
    }

    writeIsInteracting(isInteracting: boolean) {
        this.writeUint32(516, isInteracting ? 1 : 0);
    }

    writeInteractionDir(dir: [number, number, number]) {
        this.writeFloat32x3(528, dir);
    }

    writeInteractionMode(mode: number) {
        this.writeUint32(520, mode);
    }

    writeColliderFriction(friction: number) {
        this.writeFloat32(524, friction);
    }

    writeMaxStableParticleSpeed(maxStableParticleSpeed: number, maxStableParticleDisplacement: number) {
        this.maxStableParticleUpload[0] = maxStableParticleSpeed;
        this.maxStableParticleUpload[1] = maxStableParticleSpeed * maxStableParticleSpeed;
        this.maxStableParticleUpload[2] = maxStableParticleDisplacement;
        this.maxStableParticleUpload[3] = maxStableParticleDisplacement * maxStableParticleDisplacement;
        this.device.queue.writeBuffer(
            this.buffer,
            544,
            this.maxStableParticleUpload,
        );
    }

    writeColliderSdfGridScale(gridScale: [number, number, number]) {
        this.writeFloat32x3(560, gridScale);
    }

    writeColliderSdfCellSize(cellSize: [number, number, number]) {
        this.writeFloat32x3(576, cellSize);
    }

    writeColliderSdfValid(valid: boolean) {
        this.writeUint32(588, valid ? 1 : 0);
    }

    writeColliderWorldMinCoords(min: [number, number, number]) {
        this.writeFloat32x3(592, min);
    }

    writeColliderWorldMaxCoords(max: [number, number, number]) {
        this.writeFloat32x3(608, max);
    }

    writeColliderSdfMaxCellSize(maxCellSize: number) {
        this.writeFloat32(620, maxCellSize);
    }

    private writeInteractionStrengthDelta() {
        this.writeFloat32(700, this.interactionStrength * this.simulationTimestepS);
    }

    private writeTimestepDerivedValues() {
        const timestep = this.simulationTimestepS;
        const upload = this.timestepDerivedUpload;

        upload[0] = 0;
        upload[1] = 0;
        upload[2] = -GRAVITATIONAL_ACCELERATION_M_PER_S2 * timestep;
        upload[3] = this.interactionStrength * timestep;
        upload[4] = 4 * this.invGridCellDims[0];
        upload[5] = 4 * this.invGridCellDims[1];
        upload[6] = 4 * this.invGridCellDims[2];
        upload[7] = 1;
        upload[8] = -4 * timestep * this.invGridCellDimsSquared[0] * FIXED_POINT_SCALE * INVERSE_PARTICLE_DENSITY;
        upload[9] = -4 * timestep * this.invGridCellDimsSquared[1] * FIXED_POINT_SCALE * INVERSE_PARTICLE_DENSITY;
        upload[10] = -4 * timestep * this.invGridCellDimsSquared[2] * FIXED_POINT_SCALE * INVERSE_PARTICLE_DENSITY;
        upload[11] = -timestep * FIXED_POINT_SCALE * INVERSE_PARTICLE_DENSITY;
        this.device.queue.writeBuffer(this.buffer, 688, upload);
    }

}
