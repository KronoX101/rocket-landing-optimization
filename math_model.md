# Mathematical Model for 3D Rocket Landing and Optimization

This document summarizes the mathematical models and optimization problems implemented in the project.

---

## 1. State, Control, and Parameters

### State vector

```
x(t) = [x(t), y(t), z(t), v�x (t), vᵧ(t), vᵤ(t), m(t)]ᵀ ∈ ℝ⁷
```

where (x,y,z) is position in an inertial frame with +z upward, (vₓ,vᵧ,vᵤ) is velocity, and m is vehicle mass.

### Control (thrust)

```
T(t) = [Tₓ(t), Tᵧ(t), Tᵤ(t)]ᵀ ∈ ℝ³
```

T(t) = thrust vector [N]

### Physical constants

| Symbol | Meaning | Nominal value |
|--------|---------|---------------|
| g | gravity | 9.81 m/s² |
| g₀ | standard gravity | 9.81 m/s² |
| Iₛₚ | specific impulse | 300 s |
| Cₐ | drag coefficient | 0.5 |
| A | reference area | 10 m² |
| ρ₀ | sea-level density | 1.225 kg/m³ |
| H | scale height | 8500 m |

### Mass and thrust bounds

```
mₐᵣᵧ ≤ m(t) ≤ m₀
Tₘᵢₙ ≤ ‖T(t)‖₂ ≤ Tₘₐₓ
```

with Tₘᵢₙ = 6000 N, Tₘₐₓ = 15000 N, and mₐᵣᵧ = 500 kg.

---

## 2. Continuous-Time Dynamics

The dynamics follow the powered-descent model from Açıkmeşe & Ploen (2007) with atmospheric drag:

### 2.1 Atmospheric density (exponential model)

```
ρ(z) = ρ₀ exp(-z/H),  z ≥ 0
```

### 2.2 Velocity and drag

Let v(t) = [vₓ, vᵧ, vᵤ]ᵀ, ‖v‖ = √(vₓ² + vᵧ² + vᵤ²), and

```
v̂ = v/‖v‖  if ‖v‖ > 0
v̂ = 0      if ‖v‖ = 0
```

Aerodynamic drag is modeled as:

```
D(t) = -(1/2) ρ(z) Cₐ A ‖v‖² v̂ ∈ ℝ³
```

### 2.3 Equations of motion

```
ẋ = vₓ
ẏ = vᵧ
ż = vᵤ
v̇ₓ = (Tₓ + Dₓ)/m
v̇ᵧ = (Tᵧ + Dᵧ)/m
v̇ᵤ = (Tᵤ + Dᵤ)/m - g
ṁ = -‖T‖₂/(Iₛₚ g₀)
```

Compactly:

```
ẋ(t) = f(x(t), T(t))
```

where f encodes the equations above.

### 2.4 Thrust pointing (gimbal) constraint

Let eᵤ = [0, 0, 1]ᵀ. The thrust vector must lie within a cone of half-angle θₘₐₓ = 30° about the +z-axis:

```
T(t)ᵀ eᵤ ≥ ‖T(t)‖₂ cos(θₘₐₓ)
```

---

## 3. Boundary Conditions

### 3.1 Initial conditions (final descent phase)

```
x(0) = 0,  y(0) = 0,  z(0) = 1000 m
vₓ(0) = 5 m/s,  vᵧ(0) = 0,  vᵤ(0) = -50 m/s
m(0) = m₀ = 1000 kg
```

### 3.2 Terminal constraints (soft landing)

```
x(tₓ) = 0,  y(tₓ) = 0,  z(tₓ) = 0
vₓ(tₓ) = 0,  vᵧ(tₓ) = 0,  vᵤ(tₓ) = 0
m(tₓ) ≥ mₐᵣᵧ
```

---

## 4. Classical Numerical Methods

### 4.1 Bisection for Constant Thrust

We seek a constant vertical thrust Tᵤ* such that the terminal vertical velocity is nearly zero.

**Control law:**

```
T(t) = [0, 0, Tᵤ]ᵀ,  0 ≤ t ≤ tₓ
```

**Define scalar function:**

```
φ(Tᵤ) ≜ vᵤ(tₓ; Tᵤ)
```

where vᵤ(tₓ; Tᵤ) is obtained by integrating the dynamics to time tₓ with constant Tᵤ.

**Bisection algorithm:**

1. Initialize Tₗₒw = Tₘᵢₙ, Tₕᵢgₕ = Tₘₐₓ
2. Iterate:
   - Tₘᵢₐ = (Tₗₒw + Tₕᵢgₕ)/2
   - Simulate, compute e = φ(Tₘᵢₐ)
   - If |e| < ε, accept Tᵤ* = Tₘᵢₐ
   - If e < 0 (still descending): set Tₗₒw = Tₘᵢₐ
   - If e > 0 (ascending): set Tₕᵢgₕ = Tₘᵢₐ

This is a root-finding problem for φ(Tᵤ) = 0 using bisection.

### 4.2 RK4 Time Integration

Given ẋ = f(t,x), the classical fourth-order Runge–Kutta method with step h is:

```
k₁ = f(tₖ, xₖ)
k₂ = f(tₖ + h/2, xₖ + (h/2)k₁)
k₃ = f(tₖ + h/2, xₖ + (h/2)k₂)
k₄ = f(tₖ + h, xₖ + h k₃)
xₖ₊₁ = xₖ + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)
```

In this project h = Δt is typically 0.02 s to 0.05 s.

### 4.3 Cubic Spline Thrust Parameterization

We parameterize the time-varying thrust as a cubic spline through N knot points at times:

```
0 = t₀ < t₁ < ... < tₙ₋₁ = tₓ
```

At each knot tᵢ we specify:

```
Tᵢ = [Tₓ,ᵢ, Tᵧ,ᵢ, Tᵤ,ᵢ]ᵀ
```

and for each component we construct a C²-continuous cubic spline:

```
Tₓ(t) = Sₓ(t),  Tᵧ(t) = Sᵧ(t),  Tᵤ(t) = Sᵤ(t)
```

where S are piecewise-cubic polynomials with clamped boundary conditions implemented via SciPy's `CubicSpline`.

### 4.4 Classical Optimization Objective

For a given thrust profile T(t) and corresponding terminal state x(tₓ), define:

```
rₓ = [x(tₓ), y(tₓ), z(tₓ)]ᵀ
vₓ = [vₓ(tₓ), vᵧ(tₓ), vᵤ(tₓ)]ᵀ
mₓ = m(tₓ)
```

The cost used for spline-based optimization is:

```
Jclassical = 1000 ‖rₓ‖₂² + 1000 ‖vₓ‖₂² + (m₀ - mₓ)
```

where the first two terms penalize position and velocity error at touchdown, and the last term penalizes fuel consumption.

The decision variables are the thrust values at the knot points:

```
θ = [T₀ᵀ, T₁ᵀ, ..., Tₙ₋₁ᵀ]ᵀ ∈ ℝ³ᴺ
```

We solve the unconstrained, bound-limited problem:

```
min Jclassical(θ)
 θ
```

with simple component-wise bounds on Tₓ,ᵢ, Tᵧ,ᵢ, Tᵤ,ᵢ, using L-BFGS-B.

### 4.5 Monte Carlo Uncertainty Quantification

We model uncertainties as Gaussian perturbations:

**Initial position:**

```
δr₀ ~ N(0, diag(5², 5², 20²)) m
```

**Initial velocity:**

```
δv₀ ~ N(0, diag(2², 2², 5²)) m²/s²
```

**Initial mass:**

```
δm₀ ~ N(0, 30²) kg²
```

**Thrust noise (per time step):**

```
Tnoisy(t) = (1 + η(t)) Tnom(t),  where η(t) ~ N(0, 0.05²)
```

Disturbance "wind" is modeled as random horizontal force with magnitude proportional to a Gaussian random variable (see code for details).

For each trial i we generate perturbed initial conditions:

```
r₀⁽ⁱ⁾ = r₀ + δr₀⁽ⁱ⁾
v₀⁽ⁱ⁾ = v₀ + δv₀⁽ⁱ⁾
m₀⁽ⁱ⁾ = max(mₐᵣᵧ, m₀ + δm₀⁽ⁱ⁾)
```

simulate to obtain rₓ⁽ⁱ⁾, vₓ⁽ⁱ⁾, and compute:

```
eₚₒₛ⁽ⁱ⁾ = ‖rₓ⁽ⁱ⁾‖₂
eᵥₑₗ⁽ⁱ⁾ = ‖vₓ⁽ⁱ⁾‖₂
```

Define success indicator:

```
Iₛᵤcc⁽ⁱ⁾ = 1  if eₚₒₛ⁽ⁱ⁾ < 5 m and eᵥₑₗ⁽ⁱ⁾ < 2 m/s
Iₛᵤcc⁽ⁱ⁾ = 0  otherwise
```

The estimated success rate is:

```
p̂ₛᵤcc = (1/Ntrials) Σ Iₛᵤcc⁽ⁱ⁾
```

---

## 5. Convex Optimization (SOCP) Formulation

Following Açıkmeşe & Ploen (2007) and Blackmore et al. (2010), we introduce the key change of variables:

**Log mass:**

```
z(t) = ln(m(t))
```

**Acceleration control:**

```
u(t) = T(t)/m(t) ∈ ℝ³
```

**Thrust-per-mass magnitude:**

```
Γ(t) = ‖T(t)‖₂/m(t) ∈ ℝ₊
```

In these variables, the translational dynamics become **affine**:

```
v̇(t) = gvec + u(t)
```

with gvec = [0, 0, -g]ᵀ. Mass dynamics become:

```
ż(t) = -Γ(t)/(Iₛₚ g₀)
```

### 5.1 Discretization

We discretize over k = 0, ..., N with step Δt:

```
tₖ = k Δt,  Δt = constant
```

**Decision variables:**

- rₖ ∈ ℝ³ — position at step k
- vₖ ∈ ℝ³ — velocity at step k
- zₖ ∈ ℝ — log mass at step k
- uₖ ∈ ℝ³ — acceleration control at step k
- Γₖ ∈ ℝ — thrust-per-mass magnitude at step k

**Discrete dynamics** (trapezoidal for position, forward Euler for others):

```
rₖ₊₁ = rₖ + (Δt/2)(vₖ + vₖ₊₁)
vₖ₊₁ = vₖ + Δt(gvec + uₖ)
zₖ₊₁ = zₖ - (Δt/(Iₛₚ g₀))Γₖ
```

for k = 0, ..., N-1

### 5.2 Convex Constraints

**1. SOC constraint (thrust cone):**

```
‖uₖ‖₂ ≤ Γₖ,  k = 0, ..., N-1
```

This is a standard second-order cone constraint.

**2. Thrust magnitude bounds (conservative DCP-compliant form):**

To remain DCP-compliant in CVXPY, we use constant bounds based on mass range:

```
Γₘᵢₙ = Tₘᵢₙ/m₀
Γₘₐₓ = Tₘₐₓ/mₐᵣᵧ
```

Then, for all k:

```
Γₘᵢₙ ≤ Γₖ ≤ Γₘₐₓ
```

This ensures ‖Tₖ‖₂ = mₖ Γₖ remains between conservative lower and upper limits throughout the descent.

**3. Mass and altitude bounds:**

```
zₖ ≥ ln(mₐᵣᵧ),  k = 0, ..., N
zₖ ≤ ln(m₀),    k = 0, ..., N  (implicitly, via dynamics)
(rₖ)ᵤ ≥ 0,       k = 0, ..., N
```

**4. Boundary conditions:**

```
r₀ = r(0),  v₀ = v(0),  z₀ = ln(m₀)
rₙ = rtarget = [0, 0, 0]ᵀ
vₙ = vtarget = [0, 0, 0]ᵀ
```

### 5.3 Objective

The convex optimization objective is to **maximize final log mass** (equivalently, minimize fuel consumed):

```
maximize zₙ
```

subject to all constraints above. Since mₙ = exp(zₙ), this is equivalent to maximizing final mass.

### 5.4 Recovery of Physical Quantities

Given an optimal solution {rₖ, vₖ, zₖ, uₖ, Γₖ}, we recover:

```
mₖ = exp(zₖ)
```

and the thrust vectors:

```
Tₖ = mₖ uₖ,  k = 0, ..., N-1
```

For plotting, the last sample is padded as Tₙ = Tₙ₋₁.

---

## 6. Energy Accounting

For a given state x = [x, y, z, vₓ, vᵧ, vᵤ, m]ᵀ we define:

**Kinetic energy:**

```
EKE = (1/2) m ‖v‖₂²
```

**Potential energy** (relative to z = 0):

```
EPE = m g z
```

The **total mechanical energy** is:

```
Etot = EKE + EPE
```

In the simulations we monitor Etot(t) to validate that the numerical integration and drag model behave consistently (large artificial energy drift would indicate numerical issues).

---

## 7. Experiment-Specific Quantities

Several experiments vary initial or target conditions and measure fuel and accuracy metrics:

**Fuel consumption:**

```
Jfuel = m(0) - m(tₓ)
```

**Landing error:**

```
eₚₒₛ = ‖r(tₓ)‖₂
eᵥₑₗ = ‖v(tₓ)‖₂
```

**Discretization error metric (convex study):**

For each N, we use:

```
Eₙ = ‖rₙ‖₂ + ‖vₙ‖₂
```

as a scalar measure of terminal constraint satisfaction under a given time discretization.

These scalar metrics are then used to generate bar charts and heatmaps for comparison between classical and convex methods.