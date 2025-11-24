# WebGPU snow sim
This project contains a snow simulation engine using the **material point method** (**MPM**) algorithm.

## Code
This is a SvelteKit project with a simulation powered by WebGPU, so you can [view this demo online](https://luoluobuli.github.io/WebGPU-MPM-based-Snow-Simulation/) so long as your browser supports WebGPU!

### Local development
After cloning this repository:
1. Setup: `npm i` (Node.js) or `deno i` (Deno)
1. Run: `npm run dev` or `deno task dev`

## Motivation
Existing simulation systems in 3D tools like Blender are somewhat limited. There are simulations for mesh-based solids (cloth/rigid body) and simulations for particle-based solids (soft body), liquid (fluid), and gas (smoke). However, at present, there isn't exactly a solution for fragile materials like snow or jelly that only sometimes stay together and sometimes break apart.

The closest practical solution there is is to pre-fracture a mesh, sometimes manually if there are not many pieces or otherwise using a "cell fracture" mechanism which tends to be pretty easy to spot visually. In movies, simulations have evolved to use certain techniques which better support semi-solid materials. For this project, we'll focus on the "material point method", which is an algorithm often leveraged for snow, as with *Frozen* in 2013. One goal of this project is to implement and describe this technique in an accessible and customizable way, both to demonstrate its benefits as well as lay the groundwork for engine implementations for artistic use in tools like Blender.

An additional point of interest is animated characters. Even in industry tooling, characters made of a simulatable material like sand, snow, or water often simply consist of a mesh styled to look like the material, with a particle system on its surface. A large amount of manual labor is necessary to model simple interactions with other characters or meshes in a believable way, such as controllers to decide when and where to fracture the mesh into pre-factured pieces or blend between the mesh and a particle system. We'd like to investigate how easily a simulation can achieve similar results; importing meshes and animations to the scene, while maintaining ease of artistic control, is an additional point of interest.

## Implementation
Let's walk through the implementation of the simulation from the ground up.

> [!tip]
> Think something can be explained better? Open a pull request! :)

### Particles and external forces
We'll start with a simple particle simulation. We'll spawn a certain number of particles making up our material. They each have their own position $\mathbf x$ in meters $\text m$, velocity $\mathbf v$ in meters per second $\dfrac {\text m}{\text s}$, and mass $m$ in kilograms $\text{kg}$.
```wgsl
struct ParticleData {
    position: vec3f,
    velocity: vec3f,
    mass: f32,
}
```
Every timestep (of a fixed duration $\Delta t$ in seconds $\text s$), we'll calculate the force $\mathbf f$ (in newtons $\text N$) on each particle. For now, the only force we'll calculate is a constant gravity force $\mathbf f = \mathbf f_g = m \cdot \langle 0, 0, -9.81 \rangle \text{ N}$ (where +z is our "up" vector).

<img src="./docs/particle-force.png" alt="particle with force" height="300" />

> [!note]
> Lowercase $\mathbf f$ for force here. Uppercase $\mathbf F$ is used for the deformation matrix later on!

Using [Newton's second law of motion](https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion), we can obtain the acceleration on each particle in meters per second per second $\dfrac {\text m}{\text s^2}$: $\mathbf f = m \mathbf a \implies \mathbf a = \dfrac{\mathbf f}{ m} = \langle 0, 0, -9.81 \rangle \text{ }\dfrac {\text m}{\text s^2}$. (If we wanted, we could also add some user-controlled forces so we can grab and throw the particles from the UI, to make sure our simulation works with all kinds of forces.)

We now have the rate of change of position (velocity) as well as the rate of change of velocity (acceleration). To update our particles' states, we'll use [Euler integration](https://en.wikipedia.org/wiki/Euler_method), or simply multiplying the rates of change by our timestep $\Delta t$, to obtain the velocity and position of the particle in the next frame:

$$\begin{align*}
    \mathbf v_{\text{next}} &:= \mathbf v + \mathbf a \cdot \Delta t \\
    \mathbf x_{\text{next}} &:= \mathbf x + \mathbf v \cdot \Delta t
\end{align*}$$

If we do a timestep or a few every frame, our particles should now fall downward!

### Material point method
Now our particles are all moving independently of each other, so they're all acting like separate objects rather than a continuous material.

**Material point method** (**MPM**) is a technique that will help us model the *cohesion* of our material. With MPM, we'll treat each particle as a **material point**, or a sample of how our imagined continuous, deformable material is moving or deformed at a certain point in space.

MPM will also introduce a *grid* that we'll modify with our forces, rather than the particles directly. Notably, grid cells are built to not have gaps in between them... which makes a grid strategy fitting for a continuous material! The main drivers of the simulation are still the particles, though, which gives us the flexibility to move around freely without being constrained by the grid.

We can divide the main MPM algorithm into 4 steps:

1. **Particle-to-grid (P2G).** Transfer the momentum $\mathbf p = m\mathbf v$ and mass of each particle to grid cells that are near the particle. To do this transfer, we'll weight the particle's *influence* on those grid cells based on the distance of the cell's center from the particle.
1. **Grid update.** Update each grid cell's momentum and mass using our external forces.
1. **Grid-to-particle (G2P).** Transfer the momentum of each grid cell back to the particles in its vicinity.
1. **Grid clear.** Zero out all the grid cell momentums and masses, so the next frame can do this process over again.

Before we jump in, let's construct our MPM grid first.
```wgsl
struct GridUniforms {
    grid_min_coords: vec3f, // bottom-left-front corner of the grid
    grid_cell_dims: vec3f, // length/width/height of each cell
    n_cells_per_grid_axis: vec3u, // number of cells in each axis

    fixed_point_scale: f32, // fixed-point scale factor
}

struct CellData {
    momentum_x: atomic<i32>,
    momentum_y: atomic<i32>,
    momentum_z: atomic<i32>,
    mass: atomic<i32>,
}
```
Note the use of atomics here as well as the inclusion of a fixed-point scale factor. We'll come back to those after describing the particle-to-grid step.

> [!tip]
> We started seeing some nice-looking snow breakage effects at a grid cell resolution of 128 in a 4-meter-large grid and 50 000+ particles! If your grid resolution is too low, your snow will move more slowly and deform more like soap bubbles.

#### Particle-to-grid
First, we need to transfer the momentum and mass of each particle to grid cells that are near the particle. To do this transfer, we'll weight the particle's *influence* on those grid cells based on the distance of the cell's center from the particle.

We'll need to calculate which grid cell $\verb|cell_number: vec3u|$ each particle belongs to, as well as the fractional positions in the cell (how far along in each direction the particle is within the cell, as a percentage) $\verb|frac_pos: vec3f|$. This can be done knowing the position $\verb|grid_min_coords: vec3f|$ where the grid starts, as well as the size $\verb|grid_cell_dims: vec3f|$ of each cell in each direction $x,y,z$ (which we'll write as a subscript $a$ here):

$$\begin{align*}
    \verb|cell_number|_a &:= \left\lfloor \frac{\mathbf x_a - \verb|grid_min_coords|_a}{\verb|grid_cell_dims|_a} \right\rfloor \\
    \verb|frac_pos|_a &:= \dfrac{\mathbf x_a - \verb|grid_min_coords|_a - \verb|cell_number|_a \cdot \verb|grid_cell_dims|_a}{\verb|grid_cell_dims|_a}
\end{align*}$$

![mpm grid-to-particle](./docs/mpm-grid-to-particle.png)

For the weights themselves, we'll use a computationally cheap formula for the weights, a **quadratic B-spline** based on the particle's fractional position in the cell.

To do this, we'll want to take the 27 nearest cells to the particle (or the 3 nearest *2D planes* of cells in each axis). Let's then calculate the amount of influence $\verb|weight|_{b, a}$ on the neighboring planes, where $a$ is the axis and $b$ is how many planes over it is in that axis, relative to the particle. We'll store these in $\verb|weight: array<vec3f, 3>|$.

> [!tip]
> For example, in the diagram above, $\verb|weight|_{0, y}$ is the weight for the column of cells containing the particle, $\verb|weight|_{-1, y}$ is the weight for the column of cells to the *left*, and $\verb|weight|_{1, y}$ is the weight for the column of cells to the *right*.

$$\begin{align*}
    \verb|weight|_{b, a} &:= 0.5 \cdot (1 - \verb|frac_pos|_a)^2 \\
    \verb|weight|_{b, a} &:= 0.75 - (\verb|frac_pos|_a - 0.5)^2 \\
    \verb|weight|_{b, a} &:= 0.5 \cdot (\verb|frac_pos|_a)^2
\end{align*}$$

(All planes farther than 1 plane away—that is, $b < -1$ or $b > 1$—are assigned a weight of $0$.)

For a single plane of cells, these formulae result in the following weights based on the particle's distance from that plane's center in each axis:

[![graph of quadratic B-spline weights based on particle distance from the center of the plane of cells. The weight peaks at 0.75 and then smoothly decreases in both directions to 0.](./docs/quadratic-b-spline.png)](https://www.desmos.com/calculator/onwvmn9wtf)
*(click to open in Desmos)*

This should give us 3 weight values for each cell (1 for each axis), which we can multiply together to obtain the overall weight $w$ of the cell.

Finally, we can calculate the amount of momentum and mass to transfer to each grid cell:

$$\begin{align*}
    \verb|grid_cell_momentum| &\verb| += | w \cdot \mathbf p \\
    \verb|grid_cell_mass| &\verb| += | w \cdot m
\end{align*}$$

Sum up these values for every particle, and we'll be ready to apply forces!

##### WebGPU atomics
Since we're working with the GPU here, we'll handle one particle per thread in a compute shader. The grid and particle data is handled in `storage` memory, shared between all threads. However, we'll run into a problem if we run this algorithm as written: we'll need to read from and write to the same grid cells from many different threads, resulting in a race condition.

Recall in our `GridCell` struct we used `atomic<i32>`s to store the momentum and mass. Atomics ensure that only one thread will read, write, or read-and-write to a memory location at a time. Since we're accumulating mass and momentum here, we'll use the `atomicAdd` function to read-and-write to the grid cell without running into conflicts.

Note that WebGPU devices won't allow floating-point-type atomics like `atomic<f32>` unless it is requested and it is available on the device. To handle this, we'll use **fixed-point arithmetic**, where we multiply our floats by a constant factor, like $1000$, before converting into an `i32` and passing it to the atomic function.

All in all, our algorithm looks something like:
```wgsl
let particle = &particle_data[thread_index];

let mass = (*particle).mass;
let momentum = (*particle).vel * mass;
let pos = (*particle).pos;

let cell_number = vec3i((pos - grid_uniforms.grid_min_coords) / grid_uniforms.grid_cell_dims);
let fractional_pos = pos - grid_uniforms.grid_min_coords - vec3f(cell_number) * grid_uniforms.grid_cell_dims;

for (var offset_z = -1i; offset_z <= 1i; offset_z++) {
    for (var offset_y = -1i; offset_y <= 1i; offset_y++) {
        for (var offset_x = -1i; offset_x <= 1i; offset_x++) {
            let cell_index = linearizeCellIndex(cell_number + vec3i(offset_x, offset_y, offset_z));
            let cell = &grid_data[cell_index];

            let weights = calculateQuadraticBSplineCellWeights(fractional_pos); // see above
            let weight = weights[u32(offset_x + 1)].x * weights[u32(offset_y + 1)].y * weights[u32(offset_z + 1)].z;

            let momentum_contribution = momentum * weight * grid_uniforms.fixed_point_scale;
            let mass_contribution = mass * weight * grid_uniforms.fixed_point_scale;

            atomicAdd(&(*cell).momentum_x, i32(momentum_contribution.x));
            atomicAdd(&(*cell).momentum_y, i32(momentum_contribution.y));
            atomicAdd(&(*cell).momentum_z, i32(momentum_contribution.z));
            atomicAdd(&(*cell).mass, i32(mass_contribution));
        }
    }
}
```

#### Grid update
In this step, we'll apply forces to the grid cells based on their stored momentum and mass. If we already know what those forces $\mathbf f$ are, this step is easy: we just need to add the force to the momentum, since it turns out that $\dfrac{\mathrm d\mathbf p}{\mathrm dt} = \mathbf f$, which means we can Euler-integrate using:

$$\mathbf p_\text{next} := \mathbf p + \mathbf f \cdot \Delta t$$

And that's it!

```wgsl
let cell = &cell_data[thread_index];

let mass = atomicLoad(&(*cell).mass);


let force = vec3f(0, 0, -9.81) * mass;
let momentum_contrib = force * uniforms.simulation_timestep * uniforms.fixed_point_scale;

atomicAdd(&(*cell).momentum_x, i32(momentum_contrib.x));
atomicAdd(&(*cell).momentum_y, i32(momentum_contrib.y));
atomicAdd(&(*cell).momentum_z, i32(momentum_contrib.z));
```

#### Grid-to-particle
In this step, we're going to accumulate the new momentums onto the particles based on the grid cells, and then we'll update the particle's position based on that momentum.

To do this, we're going to use our B-spline weights from before as well as the grid loop to determine the influence each *grid cell* now has on the current *particle*. After we've added up the velocity contributions from all the grid cells, we can then update the position!

```wgsl
let particle = &particle_data[thread_index];

let mass = (*particle).mass;
let momentum = (*particle).vel * mass;
let pos = (*particle).pos;

let pos_from_grid_min = pos - grid_uniforms.grid_min_coords;
let cell_number = vec3i(pos_from_grid_min / grid_uniforms.grid_cell_dims);
let fractional_pos = pos_from_grid_min - vec3f(cell_number) * grid_uniforms.grid_cell_dims;

let weights = calculateQuadraticBSplineCellWeights(fractional_pos); // see above

var new_particle_velocity = vec3f(0);
for (var offset_z = -1i; offset_z <= 1i; offset_z++) {
    for (var offset_y = -1i; offset_y <= 1i; offset_y++) {
        for (var offset_x = -1i; offset_x <= 1i; offset_x++) {
            let cell_index = linearizeCellIndex(cell_number + vec3i(offset_x, offset_y, offset_z));
            let cell = &grid_data[cell_index];

            let weight = weights[u32(offset_x + 1)].x
                * weights[u32(offset_y + 1)].y
                * weights[u32(offset_z + 1)].z;

            let grid_momentum = vec3f(
                atomicLoad(&(*cell).momentum_x),
                atomicLoad(&(*cell).momentum_y),
                atomicLoad(&(*cell).momentum_z),
            ) / grid_uniforms.fixed_point_scale;
            let grid_mass = atomicLoad(&(*cell).mass);


            new_particle_velocity += grid_momentum / grid_mass * weight;
        }
    }
}

(*particle).vel = new_particle_velocity;
(*particle).pos += (*particle).vel * uniforms.simulation_timestep;
```

If we'd like, we can also add some boundary conditions here to handle any particles that end up leaving the grid.

#### Grid clear
Since we add together all the momentums and masses together in the particle-to-grid step, we need to zero out the grid cells before we start accumulating them again for the next timestep. We'll just use some simple `atomicStore` calls:

```wgsl
let cell = &cell_data[thread_index];

atomicStore(&(*cell).momentum_x, 0);
atomicStore(&(*cell).momentum_y, 0);
atomicStore(&(*cell).momentum_z, 0);
atomicStore(&(*cell).mass, 0);
```

#### Next steps
After running the 4 steps repeatedly in a simulation loop, our MPM implementation is complete! Notably, if you add forces other than gravity, you can start to notice the particles bunching up. The velocities of particles will influence nearby particles, making the material look more cohesive, with a goopy or stringy look, especially if you add forces other than gravity.

At this point, though, we've only modeled external forces. In fact, if we only have gravity, then we'll just see all the particles fall straght down through the material, which isn't very interesting and doesn't really model any material all that well.

Let's make things more interesting with some *internal forces*!

### Internal forces
Recall how our *material points* represent samples of how a continuous material is moving or deformed at each of our particles' positions. It turns out that deformation is the key to adding internal forces!

#### Deformation
The **deformation matrix** $\mathbf F$ represents the *local transformation* of the material at a given material point. By using a matrix, we have the ability to represent shearing, stretching, and rotation of the material.

> [!note]
> It'll be helpful here to know how matrices represent linear transformations! In the diagrams below, we're going to draw the deformation matrix as how it transforms the basis vectors of a coordinate system.

Deformation is a property of our particles, so let's update our `ParticleData` struct:
```wgsl
struct ParticleData {
    pos: vec3f;
    vel: vec3f;
    mass: f32;
    deformation: mat3x3f; // !! NEW
}
```

When initializing our particles, we'll want to set this to the identity matrix $\mathbf I$, representing no deformation. Over time, deformation will accumulate in this matrix as the material deforms at that point.

The first thing we'll do is **modify the particle-to-grid step** to calculate the current change in deformation (wrt time) $\dfrac{\mathrm d\mathbf F}{\mathrm dt}$ at each particle. One way to think about this change in deformation is to consider how much the material's velocity $\mathbf v_\text{material}$ varies on opposite sides of the particle.

![velocity field of antiparallel vectors that results in a shear](./docs/deformation-velocity-field.png)

We can tell that the velocity field above results in a shearing effect, where the area above the particle is moving rightward and the area below is moving leftward. To get our desired change in deformation, we can write the particle's basis vectors for the diagram on the right side, and then take their difference with the basis vectors on the left side:

$$\frac{\mathrm d\mathbf F}{\mathrm dt} = \begin{bmatrix}1 & 1 \\ 0 & 1\end{bmatrix} - \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix} = \begin{bmatrix}0 & 1 \\ 0 & 0\end{bmatrix}$$

Also note that $\mathbf F$ is not a homogeneous transformation matrix, so it doesn't encode translation. Therefore, even if the velocity vectors result in a net displacement, we only encode the stretching, shearing, and rotation in the deformation matrix:


![velocity field of non-antiparallel vectors that results in a shear](./docs/deformation-velocity-field-nonantiparallel.png)

$$\frac{\mathrm d\mathbf F}{\mathrm dt} = \begin{bmatrix}1 & 0 \\ 0.25 & 1\end{bmatrix} - \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix} = \begin{bmatrix}0 & 0 \\ 0.25 & 0\end{bmatrix}$$

We can probably intuit now that this change in deformation is somehow dependent on *how the velocity vector varies wrt position along each axis*. In calculus terms, we might say we want to take the derivative of the velocity field $\mathbf v_\text{material}$ with respect to the position $\mathbf x$. We'll call the result of this the **velocity gradient**. It turns out that the tool for differentiating a vector with respect to another vector is the **Jacobian matrix** $\mathbf J$, which is simply a matrix such that $\mathbf J_{i,j} = \dfrac{\partial\mathbf v_{\text{material},i}}{\partial\mathbf x_j}$ represents the derivative of the $i$th component of $\mathbf v_\text{material}$ with respect to the $j$th component of $\mathbf x$. For our 2D case:

$$\frac{\mathrm d\mathbf v_\text{material}}{\mathrm d\mathbf x} = \begin{bmatrix}
    \dfrac{\partial\mathbf v_{\text{material},x}}{\partial\mathbf x_x} & \dfrac{\partial\mathbf v_{\text{material},x}}{\partial\mathbf x_y}\\
    \dfrac{\partial\mathbf v_{\text{material},y}}{\partial\mathbf x_x} & \dfrac{\partial\mathbf v_{\text{material},y}}{\partial\mathbf x_y}
\end{bmatrix}$$

...which, for both of the examples above, gives us exactly the values we had written for $\dfrac{\mathrm d\mathbf F}{\mathrm dt}$!

Great! We now know how to calculate the change in deformation given a velocity field, $\dfrac{\mathrm d\mathbf F}{\mathrm dt} = \dfrac{\mathrm d\mathbf v_\text{material}}{\mathrm d\mathbf x}$. One little problem, though: we don't exactly have a continuous velocity field to differentiate, but a set of discrete velocities at the center of each grid cell. We need some way to interpolate those velocities into a continuous field.

Let's return to our 3D simulation and try writing out the velocity field $\mathbf v_\text{material}(\mathbf x)$ as a function of $\mathbf x$ so we know what we should be trying to differentiate. Recall that, in the grid-to-particle step, we analyzed how much velocity each single cell contributed to our particle's velocity. Also keep in mind that our particles are *material points* representing *samples* of a continuous material's velocity and deformation; by definition, the material's velocity $\mathbf v_\text{material}(\mathbf x)$ at any particle's position is just the velocity of that particle.

So, let's take another look at how we calculated each particle's velocity. Recall the code we used to get the weight $w$ for a specific cell and then have the cell contribute velocity to the particle:
```wgsl
let weight = weights[u32(offset_x + 1)].x
    * weights[u32(offset_y + 1)].y
    * weights[u32(offset_z + 1)].z;

// ...

new_particle_velocity += grid_velocity * weight;
```
Notably, the B-spline functions we used in the `weights` array are continuous, differentiable functions of position. And since the overall velocity is just the sum of 27 cells' contributions, any particle's resulting velocity is *also* going to be a differentiable function of position! So we can write our velocity field as a big sum of 27 cells' contributions. The derivative of that velocity field wrt position, therefore, is a big sum of 27 cells' contributions' derivatives wrt position.

As we see above, a single cell's contribution to the particle's velocity is $w \cdot \mathbf v_\text{cell}$, where $\mathbf v_\text{cell}$ is the velocity of the cell. $\mathbf v_\text{cell}$ for a given cell is constant during the entire grid-to-particle step, so we'll mainly need to worry about differentiating $w$ with respect to position. We'll need some more Jacobians for this:

$$\begin{align*}
    \frac{\mathrm dw}{\mathrm d\mathbf x} &= \begin{bmatrix}
        \dfrac{\partial w}{\partial \mathbf x_x}&
        \dfrac{\partial w}{\partial \mathbf x_y}&
        \dfrac{\partial w}{\partial \mathbf x_z}
    \end{bmatrix}
    \\
    \frac{\mathrm d[w \cdot \mathbf v_\text{cell}]}{\mathrm d\mathbf x} &= \begin{bmatrix}
        \dfrac{\partial w}{\partial \mathbf x_x}\cdot \mathbf v_\text{cell}&
        \dfrac{\partial w}{\partial \mathbf x_y}\cdot \mathbf v_\text{cell}&
        \dfrac{\partial w}{\partial \mathbf x_z}\cdot \mathbf v_\text{cell}
    \end{bmatrix}
\end{align*}$$

Let's split $w = w_x \cdot w_y \cdot w_z$ into the 3 different weights $w_x, w_y, w_z$ we multiplied to calculate it. Note that only $w_x$ will vary with $\mathbf x_x$, only $w_y$ will vary with $\mathbf x_y$, and only $w_z$ will vary with $\mathbf x_z$. So we can further rewrite our Jacobian as:

$$\begin{align*}
    \frac{\mathrm dw}{\mathrm d\mathbf x} &= \begin{bmatrix}
        \dfrac{\partial w_x}{\partial \mathbf x_x} \cdot w_y \cdot w_z &
        w_x \cdot \dfrac{\partial w_y}{\partial \mathbf x_y} \cdot w_z &
        w_x \cdot w_y \cdot \dfrac{\partial w_z}{\partial \mathbf x_z}
    \end{bmatrix}
    \\
    \frac{\mathrm d[w \cdot \mathbf v_\text{cell}]}{\mathrm d\mathbf x} &= \begin{bmatrix}
        \dfrac{\partial w_x}{\partial \mathbf x_x} \cdot w_y \cdot w_z \cdot \mathbf v_\text{cell}&
        w_x \cdot \dfrac{\partial w_y}{\partial \mathbf x_y} \cdot w_z \cdot \mathbf v_\text{cell}&
        w_x \cdot w_y \cdot \dfrac{\partial w_z}{\partial \mathbf x_z} \cdot \mathbf v_\text{cell}
    \end{bmatrix}
\end{align*}$$

Our B-spline functions were piecewise, so their derivatives are also piecewise. Let's differentiate the 3 weight formulas we used above (with respect to position in each axis $\mathbf x_a$):

$$\frac{\mathrm d}{\mathrm d\mathbf x_a}\begin{bmatrix}
    0.5 \cdot (1 - \mathbf x_a)^2 \\
    0.75 - (\mathbf x_a - 0.5)^2 \\
    0.5 \cdot (\mathbf x_a)^2
\end{bmatrix} = \begin{bmatrix}
    \mathbf x_a - 1 \\
    -2 \cdot (\mathbf x_a - 0.5) \\
    \mathbf x_a
\end{bmatrix}$$

And at last, we now have all the pieces we need to calculate the change in the deformation gradient!

$$\mathbf F_\text{next} := \mathbf F + \frac{\mathrm d\mathbf v_\text{material}}{\mathrm d\mathbf x} \cdot \Delta t$$

```wgsl
let weights = calculateQuadraticBSplineCellWeights(fractional_pos); // see above
let weight_derivs = calculateQuadraticBSplineCellWeightDerivatives(fractional_pos); // see above

var total_velocity_gradient = mat3x3f();
for (var offset_z = -1i; offset_z <= 1i; offset_z++) {
    for (var offset_y = -1i; offset_y <= 1i; offset_y++) {
        for (var offset_x = -1i; offset_x <= 1i; offset_x++) {
            let cell_index = linearizeCellIndex(cell_number + vec3i(offset_x, offset_y, offset_z));
            let cell = &grid_data[cell_index];

            let weight = weights[u32(offset_x + 1)].x
                * weights[u32(offset_y + 1)].y
                * weights[u32(offset_z + 1)].z;

            // ...

            let grid_velocity = grid_momentum / grid_mass;

            let velocity_gradient_contrib = vec3f(
                weight_derivs[u32(offset_x + 1)].x * weight[u32(offset_y + 1)].y * weight[u32(offset_z + 1)].z,
                weight[u32(offset_x + 1)].x * weight_derivs[u32(offset_y + 1)].y * weight[u32(offset_z + 1)].z,
                weight[u32(offset_x + 1)].x * weight[u32(offset_y + 1)].y * weight_derivs[u32(offset_z + 1)].z,
            );

            total_velocity_gradient += mat3x3f(
                velocity_gradient_contrib.x * grid_velocity,
                velocity_gradient_contrib.y * grid_velocity,
                velocity_gradient_contrib.z * grid_velocity,
            );
        }
    }
}

(*particle).deformation += total_velocity_gradient * uniforms.simulation_timestep;
```

Even after all that, though, our deformation isn't actually doing anything to the simulation. Let's visualize the fruits of our labor by adding a force that comes as a direct result of deformation.

#### Elasticity and stress force
We noted before that all of our particles, with only a gravity force, are going to sink through the material, all the way to the bottom of the grid. The **stress** force will counteract this by attempting to restore the material's deformation to the original, undeformed state.
> [!note]
> TBD

#### Plasticity
> [!note]
> TBD

#### Hardening
> [!note]
> TBD

### Rigid colliders
> [!note]
> TBD

### Position-based material point method (PBMPM)
> [!note]
> TBD

### Raymarching
> [!note]
> TBD

## Resources
1. **[Breakpoint](https://github.com/danieljgerhardt/Breakpoint).** A DirectX implementation of 3D PBMPM with mixed material types, along with a mesh shading renderer.
1. **[GPUMPM](https://github.com/kuiwuchn/GPUMPM).** A CUDA implemenation of MPM, associated with the 2019 paper ***[GPU optimization of material point methods](https://dl.acm.org/doi/10.1145/3272127.3275044)***.
1. **[PB-MPM](https://github.com/electronicarts/pbmpm).** The original WebGPU proof-of-concept implementation of 2D PBMPM with various material types, associated with the original 2024 paper introducing PBMPM, ***[A Position Based Material Point Method](https://media.contentapi.ea.com/content/dam/ea/seed/presentations/seed-siggraph2024-pbmpm-paper.pdf)***. *(although internal forces seem to be broken at the moment, as of 2025-11-21...)*
1. ***[Principles towards Real-Time Simulation of Material Point Method on Modern GPUs](https://arxiv.org/pdf/2111.00699)*.** A 2021 paper covering GPU optimizations of MPM.

[Project presentation slides](https://docs.google.com/presentation/d/1KzaJZwBxE9-vjqXS8KjHvptEbJPF9yljNl7gesTuuB4/edit?usp=sharing)

## Acknowledgments

### Libraries used
1. **[SASS](https://sass-lang.com/).** CSS preprocessing.
1. **[SvelteKit](https://svelte.dev/).** UI/reactivity.
1. **[THREE.js](https://threejs.org/).** gLTF/gLB loading.
1. **[Vite](https://vite.dev/).** Bundling and development environment.
1. **[wgpu-matrix](https://github.com/greggman/wgpu-matrix).** Helper classes for operations involving WGSL builtins on the CPU.
