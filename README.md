## Numerical Optimization of Rocket Landing Trajectories

This project implements and compares **classical numerical methods** and **modern convex optimization** techniques for the 3D powered descent guidance problem of a reusable rocket stage. The code is written in modern Python (3.9+) with explicit type hints and is suitable for academic use, teaching, and reproducible research.

The dynamics model includes 3D translational motion, variable mass due to propellant consumption, and atmospheric drag. Two solution strategies are implemented:

- **Method A – Classical:** Bisection root-finding, high-order RK4 integration, cubic spline thrust optimization, and Monte Carlo robustness analysis.
- **Method B – Convex:** Lossless convexification of the powered descent guidance problem and solution via **Second-Order Cone Programming (SOCP)** using **CVXPY**.

Key references:

- Meditch, J.S., *On the Problem of Optimal Thrust Programming for a Lunar Soft Landing*, IEEE TAC, 1964.
- Açıkmeşe, B., and Ploen, S., *Convex Programming Approach to Powered Descent Guidance for Mars Landing*, JGCD, 2007.
- Blackmore, L., Açıkmeşe, B., and Scharf, D.P., *Minimum-Effort Guidance for Planetary Landing Using Convex Optimization*, JGCD, 2010.

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Usage

Run both solvers, all experiments, and generate figures to `outputs/`:

```bash
python main.py --method both --experiments all --save-animations
```

Run only the convex solver without experiments:

```bash
python main.py --method convex --experiments none
```

Figures (PNG + PDF) and animations (MP4) are written to the `outputs/` directory.

### File Structure

- `dynamics.py` – 3D rocket dynamics, RK4 integrator, energy validation utilities.
- `classical_solver.py` – Bisection, spline-based thrust optimization, Monte Carlo analysis.
- `convex_solver.py` – SOCP formulation and solution using CVXPY (lossless convexification).
- `visualization.py` – Publication-quality plots and 3D animations.
- `experiments.py` – Wind, offset, initial velocity, engine failure, and discretization studies.
- `main.py` – Command-line interface that orchestrates solvers, experiments, and visualization.
- `requirements.txt` – Project dependencies.

### Expected Results (Typical)

On a modern laptop (Python 3.9+):

- Convex SOCP solve times: **50–200 ms** for \(N \approx 50\) steps.
- Monte Carlo (1000 trials) completes in **< 2 minutes**.
- Landing accuracy:
  - Classical (optimized spline): **< 2 m** position error.
  - Convex (SOCP): **< 0.5 m** position error.
- Fuel savings: Convex method typically saves **15–30%** fuel relative to constant-thrust classical strategies.

Exact numbers will depend on numerical tolerances, solver versions, and machine performance, but the qualitative result that convex optimization outperforms simple classical strategies is robust.

### Academic Notes and Extensions

This code is designed to be easy to extend. Possible directions:

- Add attitude dynamics and gimbal limits as explicit constraints.
- Incorporate more realistic atmospheric and wind models.
- Implement real-time re-planning with updated state estimates.
- Compare different SOCP solvers and discretization schemes.

All modules are documented with Google-style docstrings and use Python logging for reproducibility and diagnostics. The project is suitable as a starting point for coursework, theses, or research on optimal control for planetary landing.


