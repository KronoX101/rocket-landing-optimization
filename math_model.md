## Mathematical Model for 3D Rocket Landing and Optimization

This document summarizes the mathematical models and optimization problems implemented in the project.

---

## 1. State, Control, and Parameters

- **State vector**

\[
x(t) =
\begin{bmatrix}
x(t) \\ y(t) \\ z(t) \\ v_x(t) \\ v_y(t) \\ v_z(t) \\ m(t)
\end{bmatrix}
\in \mathbb{R}^7
\]

where \((x,y,z)\) is position in an inertial frame with \(+z\) upward, \((v_x,v_y,v_z)\) is velocity, and \(m\) is vehicle mass.

- **Control (thrust)**

\[
T(t) =
\begin{bmatrix}
T_x(t) \\ T_y(t) \\ T_z(t)
\end{bmatrix}
\in \mathbb{R}^3, \qquad
T(t) = \text{thrust vector [N]}.
\]

- **Physical constants**

| Symbol | Meaning | Nominal value |
|--------|---------|---------------|
| \(g\)  | gravity               | \(9.81\ \text{m/s}^2\) |
| \(g_0\)| standard gravity      | \(9.81\ \text{m/s}^2\) |
| \(I_{sp}\) | specific impulse | \(300\ \text{s}\) |
| \(C_d\)| drag coefficient      | \(0.5\) |
| \(A\)  | reference area        | \(10\ \text{m}^2\) |
| \(\rho_0\) | sea-level density| \(1.225\ \text{kg/m}^3\) |
| \(H\)  | scale height          | \(8500\ \text{m}\) |

- **Mass and thrust bounds**

\[
m_{\text{dry}} \le m(t) \le m_0, \qquad
T_{\min} \le \|T(t)\|_2 \le T_{\max},
\]

with \(T_{\min} = 6000\ \text{N}\), \(T_{\max} = 15000\ \text{N}\), and \(m_{\text{dry}} = 500\ \text{kg}\).

---

## 2. Continuous-Time Dynamics

The dynamics follow the powered-descent model from Açıkmeşe & Ploen (2007) with atmospheric drag:

1. **Atmospheric density** (exponential model)

\[
\rho(z) = \rho_0 \exp\!\left(-\frac{z}{H}\right), \qquad z \ge 0.
\]

2. **Velocity and drag**

Let \(v(t) = [v_x,v_y,v_z]^\top\), \(\|v\| = \sqrt{v_x^2+v_y^2+v_z^2}\), and

\[
\hat{v} =
\begin{cases}
v / \|v\|, & \|v\| > 0,\\[4pt]
0, & \|v\| = 0.
\end{cases}
\]

Aerodynamic drag is modeled as

\[
D(t) = -\tfrac{1}{2}\,\rho(z)\,C_d\,A\,\|v\|^2\,\hat{v}
    \in \mathbb{R}^3.
\]

3. **Equations of motion**

\[
\begin{aligned}
\dot{x} &= v_x,\\
\dot{y} &= v_y,\\
\dot{z} &= v_z,\\
\dot{v}_x &= \frac{T_x + D_x}{m},\\
\dot{v}_y &= \frac{T_y + D_y}{m},\\
\dot{v}_z &= \frac{T_z + D_z}{m} - g,\\
\dot{m} &= -\frac{\|T\|_2}{I_{sp}\,g_0}.
\end{aligned}
\]

Compactly:

\[
\dot{x}(t) = f\big(x(t), T(t)\big),
\]

where \(f\) encodes the equations above.

4. **Thrust pointing (gimbal) constraint**

Let \(e_z = [0,0,1]^\top\). The thrust vector must lie within a cone of half-angle \(\theta_{\max}=30^\circ\) about the +z-axis:

\[
T(t)^\top e_z \ge \|T(t)\|_2 \cos\theta_{\max}.
\]

---

## 3. Boundary Conditions

### 3.1 Initial conditions (final descent phase)

\[
\begin{aligned}
x(0) &= 0,\quad y(0) = 0,\quad z(0) = 1000\ \text{m},\\
v_x(0) &= 5\ \text{m/s},\quad v_y(0) = 0,\quad v_z(0) = -50\ \text{m/s},\\
m(0) &= m_0 = 1000\ \text{kg}.
\end{aligned}
\]

### 3.2 Terminal constraints (soft landing)

\[
\begin{aligned}
x(t_f) &= 0,\quad y(t_f) = 0,\quad z(t_f) = 0,\\
v_x(t_f) &= 0,\quad v_y(t_f) = 0,\quad v_z(t_f) = 0,\\
m(t_f) &\ge m_{\text{dry}}.
\end{aligned}
\]

---

## 4. Classical Numerical Methods

### 4.1 Bisection for Constant Thrust

We seek a constant vertical thrust \(T_z^\star\) such that the terminal vertical velocity is nearly zero:

- Control law:

\[
T(t) =
\begin{bmatrix}
0 \\ 0 \\ T_z
\end{bmatrix}, \quad 0 \le t \le t_f.
\]

- Define scalar function

\[
\phi(T_z) \triangleq v_z(t_f; T_z),
\]

where \(v_z(t_f; T_z)\) is obtained by integrating the dynamics to time \(t_f\) with constant \(T_z\).

- **Bisection algorithm**:
  - Initialize \(T_{\text{low}} = T_{\min}\), \(T_{\text{high}} = T_{\max}\).
  - Iterate:

    \[
    T_{\text{mid}} = \tfrac{1}{2}(T_{\text{low}} + T_{\text{high}}),
    \]

    simulate, compute \(e = \phi(T_{\text{mid}})\).

    - If \(|e| < \varepsilon\), accept \(T_z^\star = T_{\text{mid}}\).
    - If \(e < 0\) (still descending): set \(T_{\text{low}} = T_{\text{mid}}\).
    - If \(e > 0\) (ascending): set \(T_{\text{high}} = T_{\text{mid}}\).

This is a root-finding problem for \(\phi(T_z) = 0\) using bisection.

### 4.2 RK4 Time Integration

Given \(\dot{x} = f(t,x)\), the classical fourth-order Runge–Kutta method with step \(h\) is:

\[
\begin{aligned}
k_1 &= f(t_k, x_k),\\
k_2 &= f\big(t_k + \tfrac{h}{2},\, x_k + \tfrac{h}{2}k_1\big),\\
k_3 &= f\big(t_k + \tfrac{h}{2},\, x_k + \tfrac{h}{2}k_2\big),\\
k_4 &= f\big(t_k + h,\, x_k + h k_3\big),\\[4pt]
x_{k+1} &= x_k + \frac{h}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right).
\end{aligned}
\]

In this project \(h = \Delta t\) is typically \(0.02\ \text{s}\)–\(0.05\ \text{s}\).

### 4.3 Cubic Spline Thrust Parameterization

We parameterize the time-varying thrust as a cubic spline through \(N\) knot points at times
\[
0 = t_0 < t_1 < \dots < t_{N-1} = t_f.
\]

At each knot \(t_i\) we specify

\[
T_i =
\begin{bmatrix}
T_{x,i} \\ T_{y,i} \\ T_{z,i}
\end{bmatrix},
\]

and for each component we construct a \(C^2\)-continuous cubic spline:

\[
T_x(t) = S_x(t),\quad T_y(t) = S_y(t),\quad T_z(t) = S_z(t),
\]

where \(S_\cdot\) are piecewise-cubic polynomials with clamped boundary conditions implemented via SciPy’s `CubicSpline`.

### 4.4 Classical Optimization Objective

For a given thrust profile \(T(t)\) and corresponding terminal state \(x(t_f)\), define

\[
\begin{aligned}
r_f &= \begin{bmatrix} x(t_f) \\ y(t_f) \\ z(t_f) \end{bmatrix},\\
v_f &= \begin{bmatrix} v_x(t_f) \\ v_y(t_f) \\ v_z(t_f) \end{bmatrix},\\
m_f &= m(t_f).
\end{aligned}
\]

The cost used for spline-based optimization is

\[
J_{\text{classical}} =
1000\,\|r_f\|_2^2
 + 1000\,\|v_f\|_2^2
 + (m_0 - m_f),
\]

where the first two terms penalize position and velocity error at touchdown, and the last term penalizes fuel consumption.

The decision variables are the thrust values at the knot points:

\[
\theta =
\begin{bmatrix}
T_0^\top & T_1^\top & \dots & T_{N-1}^\top
\end{bmatrix}^\top \in \mathbb{R}^{3N}.
\]

We solve the unconstrained, bound-limited problem

\[
\min_{\theta} \; J_{\text{classical}}(\theta)
\]

with simple component-wise bounds on \(T_{x,i},T_{y,i},T_{z,i}\), using L-BFGS-B.

### 4.5 Monte Carlo Uncertainty Quantification

We model uncertainties as Gaussian perturbations:

- Initial position:

\[
\delta r_0 \sim \mathcal{N}\left(0, \operatorname{diag}(5^2, 5^2, 20^2)\right)\ \text{m}.
\]

- Initial velocity:

\[
\delta v_0 \sim \mathcal{N}\left(0, \operatorname{diag}(2^2, 2^2, 5^2)\right)\ \text{m}^2/\text{s}^2.
\]

- Initial mass:

\[
\delta m_0 \sim \mathcal{N}(0, 30^2)\ \text{kg}^2.
\]

- Thrust noise (per time step):

\[
T_{\text{noisy}}(t) = (1 + \eta(t))\,T_{\text{nom}}(t),\qquad \eta(t) \sim \mathcal{N}(0, 0.05^2).
\]

Disturbance “wind” is modeled as random horizontal force with magnitude proportional to a Gaussian random variable (see code for details).

For each trial \(i\) we generate perturbed initial conditions

\[
\begin{aligned}
r_0^{(i)} &= r_0 + \delta r_0^{(i)},\\
v_0^{(i)} &= v_0 + \delta v_0^{(i)},\\
m_0^{(i)} &= \max(m_{\text{dry}},\, m_0 + \delta m_0^{(i)}),
\end{aligned}
\]

simulate to obtain \(r_f^{(i)}, v_f^{(i)}\), and compute

\[
e^{(i)}_{\text{pos}} = \|r_f^{(i)}\|_2,\qquad
e^{(i)}_{\text{vel}} = \|v_f^{(i)}\|_2.
\]

Define success indicator

\[
\mathbb{I}^{(i)}_{\text{succ}} =
\begin{cases}
1, & e^{(i)}_{\text{pos}} < 5\ \text{m and } e^{(i)}_{\text{vel}} < 2\ \text{m/s},\\
0, & \text{otherwise}.
\end{cases}
\]

The estimated success rate is

\[
\hat{p}_{\text{succ}} =
\frac{1}{N_{\text{trials}}}
\sum_{i=1}^{N_{\text{trials}}} \mathbb{I}^{(i)}_{\text{succ}}.
\]

---

## 5. Convex Optimization (SOCP) Formulation

Following Açıkmeşe & Ploen (2007) and Blackmore et al. (2010), we introduce the key change of variables:

- **Log mass**:

\[
z(t) = \ln m(t).
\]

- **Acceleration control**:

\[
u(t) = \frac{T(t)}{m(t)} \in \mathbb{R}^3.
\]

- **Thrust-per-mass magnitude**:

\[
\Gamma(t) = \frac{\|T(t)\|_2}{m(t)} \in \mathbb{R}_+.
\]

In these variables, the translational dynamics become **affine**:

\[
\dot{v}(t) = g_{\text{vec}} + u(t),
\]

with \(g_{\text{vec}} = [0,0,-g]^\top\). Mass dynamics become

\[
\dot{z}(t) = -\frac{\Gamma(t)}{I_{sp}\,g_0}.
\]

### 5.1 Discretization

We discretize over \(k = 0,\dots,N\) with step \(\Delta t\):

\[
t_k = k \Delta t, \quad \Delta t = \text{constant}.
\]

Decision variables:

\[
\begin{aligned}
r_k &\in \mathbb{R}^3 &&\text{position at step }k,\\
v_k &\in \mathbb{R}^3 &&\text{velocity at step }k,\\
z_k &\in \mathbb{R}   &&\text{log mass at step }k,\\
u_k &\in \mathbb{R}^3 &&\text{acceleration control at step }k,\\
\Gamma_k &\in \mathbb{R} &&\text{thrust-per-mass magnitude at step }k.
\end{aligned}
\]

Discrete dynamics (trapezoidal for position, forward Euler for others):

\[
\begin{aligned}
r_{k+1} &= r_k + \tfrac{1}{2}\Delta t\,(v_k + v_{k+1}),\\
v_{k+1} &= v_k + \Delta t\,(g_{\text{vec}} + u_k),\\
z_{k+1} &= z_k - \frac{\Delta t}{I_{sp}\,g_0}\,\Gamma_k,
\end{aligned}
\quad k = 0,\dots,N-1.
\]

### 5.2 Convex Constraints

1. **SOC constraint (thrust cone)**:

\[
\|u_k\|_2 \le \Gamma_k,
\quad k = 0,\dots,N-1.
\]

This is a standard second-order cone constraint.

2. **Thrust magnitude bounds (conservative DCP-compliant form)**:

To remain DCP-compliant in CVXPY, we use constant bounds based on mass range:

\[
\Gamma_{\min} = \frac{T_{\min}}{m_0},\qquad
\Gamma_{\max} = \frac{T_{\max}}{m_{\text{dry}}}.
\]

Then, for all \(k\),

\[
\Gamma_{\min} \le \Gamma_k \le \Gamma_{\max}.
\]

This ensures \(\|T_k\|_2 = m_k \Gamma_k\) remains between conservative lower and upper limits throughout the descent.

3. **Mass and altitude bounds**:

\[
\begin{aligned}
z_k &\ge \ln m_{\text{dry}}, &&k = 0,\dots,N,\\
z_k &\le \ln m_0,            &&k = 0,\dots,N \quad (\text{implicitly, via dynamics}),\\
(r_k)_z &\ge 0,              &&k = 0,\dots,N.
\end{aligned}
\]

4. **Boundary conditions**:

\[
\begin{aligned}
r_0 &= r(0),\quad v_0 = v(0),\quad z_0 = \ln m_0,\\
r_N &= r_{\text{target}} = [0,0,0]^\top,\\
v_N &= v_{\text{target}} = [0,0,0]^\top.
\end{aligned}
\]

### 5.3 Objective

The convex optimization objective is to **maximize final log mass** (equivalently, minimize fuel consumed):

\[
\max \; z_N,
\]

subject to all constraints above. Since \(m_N = \exp(z_N)\), this is equivalent to maximizing final mass.

### 5.4 Recovery of Physical Quantities

Given an optimal solution \(\{r_k, v_k, z_k, u_k, \Gamma_k\}\), we recover

\[
m_k = \exp(z_k),
\]

and the thrust vectors

\[
T_k = m_k u_k, \qquad k = 0,\dots,N-1.
\]

For plotting, the last sample is padded as \(T_N = T_{N-1}\).

---

## 6. Energy Accounting

For a given state \(x = [x,y,z,v_x,v_y,v_z,m]^\top\) we define:

- **Kinetic energy**

\[
E_{\text{KE}} = \tfrac{1}{2} m \|v\|_2^2.
\]

- **Potential energy** (relative to \(z=0\)):

\[
E_{\text{PE}} = m g z.
\]

The **total mechanical energy** is

\[
E_{\text{tot}} = E_{\text{KE}} + E_{\text{PE}}.
\]

In the simulations we monitor \(E_{\text{tot}}(t)\) to validate that the numerical integration and drag model behave consistently (large artificial energy drift would indicate numerical issues).

---

## 7. Experiment-Specific Quantities

Several experiments vary initial or target conditions and measure fuel and accuracy metrics:

- **Fuel consumption**:

\[
J_{\text{fuel}} = m(0) - m(t_f).
\]

- **Landing error**:

\[
e_{\text{pos}} = \|r(t_f)\|_2, \qquad
e_{\text{vel}} = \|v(t_f)\|_2.
\]

- **Discretization error metric (convex study)**:

For each \(N\), we use

\[
E_N = \|r_N\|_2 + \|v_N\|_2
\]

as a scalar measure of terminal constraint satisfaction under a given time discretization.

These scalar metrics are then used to generate bar charts and heatmaps for comparison between classical and convex methods.


