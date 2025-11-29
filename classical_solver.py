"""Classical numerical methods for powered descent guidance.

This module implements a set of classical numerical techniques for
landing a reusable rocket stage in three dimensions:

* Bisection method to determine a constant thrust level that yields a
  soft vertical landing (root‑finding on terminal vertical velocity).
* Time‑varying thrust optimization using cubic splines and L‑BFGS‑B to
  minimize a quadratic landing error and fuel consumption objective.
* Monte Carlo uncertainty propagation to assess robustness with respect
  to initial condition perturbations and wind / thrust noise.

These implementations provide a baseline for comparison with the convex
optimization approach described by Açıkmeşe & Ploen (2007) and
Blackmore et al. (2010).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy import interpolate, optimize

from dynamics import (
    Array,
    Rocket3D,
    RocketConfig,
    constant_thrust_profile,
    default_initial_state,
)

logger = logging.getLogger(__name__)


def profile_time(func: Callable) -> Callable:
    """Simple timing decorator for performance logging."""

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        logger.info("%s executed in %.3f s", func.__name__, dt)
        return result

    return wrapper


@dataclass
class BisectionResult:
    """Container for bisection search outcome."""

    T_opt: float
    n_iter: int
    converged: bool
    vz_final: float
    history: List[Tuple[int, float, float]]


@dataclass
class MonteCarloStats:
    """Summary statistics for Monte Carlo landing analysis."""

    success_rate: float
    mean_pos_error: float
    p95_pos_error: float
    mean_vel_error: float
    p95_vel_error: float


class ClassicalSolver:
    """Collection of classical numerical methods for rocket landing."""

    def __init__(self, rocket: Rocket3D | None = None) -> None:
        self.rocket = rocket or Rocket3D()

    # ------------------------------------------------------------------
    # Bisection for constant thrust
    # ------------------------------------------------------------------
    @profile_time
    def constant_thrust_bisection(
        self,
        t_final: float,
        vz_tol: float = 0.1,
        max_iter: int = 30,
    ) -> BisectionResult:
        """Find a constant vertical thrust that yields near‑zero vertical velocity.

        The bisection search varies a constant thrust magnitude aligned
        with the +z axis and uses RK4 simulations to evaluate the final
        vertical velocity. The objective is to satisfy ``|vz(t_final)| <
        vz_tol`` starting from :func:`default_initial_state`.

        Args:
            t_final: Final landing time horizon [s].
            vz_tol: Tolerance on terminal vertical velocity [m/s].
            max_iter: Maximum number of bisection iterations.

        Returns:
            :class:`BisectionResult` with optimal thrust and diagnostics.
        """
        cfg: RocketConfig = self.rocket.config
        T_low = cfg.T_min
        T_high = cfg.T_max

        history: List[Tuple[int, float, float]] = []

        def simulate_with_Tz(Tz: float) -> float:
            T_vec = np.array([0.0, 0.0, Tz], dtype=float)
            x0 = default_initial_state(cfg)
            t_hist, x_hist = self.rocket.simulate(
                x0=x0,
                thrust_func=constant_thrust_profile(T_vec),
                t_final=t_final,
                dt=0.02,
            )
            vz_final = float(x_hist[-1, 5])
            return vz_final

        vz_low = simulate_with_Tz(T_low)
        vz_high = simulate_with_Tz(T_high)
        if vz_low * vz_high > 0:
            logger.warning(
                "Bisection initial bracket does not straddle root: vz_low=%.3f, vz_high=%.3f",
                vz_low,
                vz_high,
            )

        converged = False
        T_opt = 0.5 * (T_low + T_high)
        vz_final = np.nan

        for i in range(max_iter):
            T_mid = 0.5 * (T_low + T_high)
            vz_mid = simulate_with_Tz(T_mid)
            history.append((i, T_mid, vz_mid))
            logger.info("Bisection iter %d: Tz=%.2f N, vz_final=%.3f m/s", i, T_mid, vz_mid)

            if abs(vz_mid) < vz_tol:
                converged = True
                T_opt = T_mid
                vz_final = vz_mid
                break

            if vz_mid < 0.0:
                # Vehicle still descending: need more thrust
                T_low = T_mid
                vz_low = vz_mid
            else:
                # Vehicle ascending at final time: too much thrust
                T_high = T_mid
                vz_high = vz_mid

        else:
            # Use midpoint of final interval as best estimate
            T_opt = 0.5 * (T_low + T_high)
            vz_final = simulate_with_Tz(T_opt)

        return BisectionResult(
            T_opt=T_opt,
            n_iter=len(history),
            converged=converged,
            vz_final=vz_final,
            history=history,
        )

    # ------------------------------------------------------------------
    # Spline‑based variable thrust optimization
    # ------------------------------------------------------------------
    def _spline_from_control_points(
        self,
        t_knots: Array,
        thrust_knots: Array,
    ) -> Callable[[float, Array], Array]:
        """Create a C^2 cubic spline thrust profile from control points.

        Args:
            t_knots: 1D array of knot times of shape (N,).
            thrust_knots: Array of shape (N, 3) with [Tx, Ty, Tz] at knots.

        Returns:
            Callable mapping (t, x) -> T(t) by spline evaluation.
        """
        t_knots = np.asarray(t_knots, dtype=float)
        thrust_knots = np.asarray(thrust_knots, dtype=float)
        if t_knots.ndim != 1 or thrust_knots.shape != (t_knots.size, 3):
            raise ValueError("Invalid spline knot shapes.")

        # Fit independent cubic splines for each thrust component.
        splx = interpolate.CubicSpline(t_knots, thrust_knots[:, 0], bc_type="clamped")
        sply = interpolate.CubicSpline(t_knots, thrust_knots[:, 1], bc_type="clamped")
        splz = interpolate.CubicSpline(t_knots, thrust_knots[:, 2], bc_type="clamped")

        def thrust_func(t: float, _x: Array) -> Array:
            T = np.array([splx(t), sply(t), splz(t)], dtype=float)
            # Enforce thrust magnitude bounds.
            T_norm = float(np.linalg.norm(T))
            if T_norm == 0:
                return np.zeros(3)
            cfg = self.rocket.config
            T_mag = np.clip(T_norm, cfg.T_min, cfg.T_max)
            return T_mag * (T / T_norm)

        return thrust_func

    @profile_time
    def variable_thrust_optimization(
        self,
        t_final: float = 40.0,
        n_knots: int = 5,
        max_iter: int = 50,
    ) -> Dict[str, Array]:
        """Optimize a spline‑parameterized thrust profile.

        The decision variables are the values of the thrust vector at a
        small set of knot times. Cubic splines interpolate these values
        to produce a smooth thrust profile. The objective is

            J = 1000 * ||r_f||^2 + 1000 * ||v_f||^2 + (m0 - m_f)

        where r_f, v_f, m_f denote terminal position, velocity, and mass.

        Args:
            t_final: Final time horizon [s].
            n_knots: Number of spline control points.
            max_iter: Maximum L‑BFGS‑B iterations.

        Returns:
            Dictionary with optimized trajectory and thrust profile.
        """
        cfg = self.rocket.config
        x0 = default_initial_state(cfg)

        # Uniformly spaced knots in time.
        t_knots = np.linspace(0.0, t_final, n_knots)

        # Initialize with constant vertical thrust near mid‑range.
        T_init_mag = 0.5 * (cfg.T_min + cfg.T_max)
        T_init = np.tile(np.array([0.0, 0.0, T_init_mag]), (n_knots, 1))

        def pack(T_knots: Array) -> Array:
            return T_knots.reshape(-1)

        def unpack(z: Array) -> Array:
            return z.reshape(-1, 3)

        def objective(z: Array) -> float:
            T_knots = unpack(z)
            thrust_func = self._spline_from_control_points(t_knots, T_knots)
            _, x_hist = self.rocket.simulate(
                x0=x0,
                thrust_func=thrust_func,
                t_final=t_final,
                dt=0.05,
            )
            xf = x_hist[-1]
            r_f = xf[0:3]
            v_f = xf[3:6]
            m_f = xf[6]

            pos_term = 1000.0 * float(np.dot(r_f, r_f))
            vel_term = 1000.0 * float(np.dot(v_f, v_f))
            fuel_term = cfg.m0 - m_f
            J = pos_term + vel_term + fuel_term

            logger.debug(
                "Objective eval: J=%.3f, ||r_f||=%.3f, ||v_f||=%.3f, m_f=%.3f",
                J,
                np.linalg.norm(r_f),
                np.linalg.norm(v_f),
                m_f,
            )
            return float(J)

        z0 = pack(T_init)

        # Bounds on each thrust component based on total magnitude limits.
        # Use generous component‑wise bounds that still enforce |T| <= T_max.
        comp_bound = cfg.T_max
        bounds = [(-comp_bound, comp_bound)] * z0.size

        logger.info(
            "Starting spline optimization with %d knots and %d variables.",
            n_knots,
            z0.size,
        )

        res = optimize.minimize(
            objective,
            z0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "disp": False},
        )

        if not res.success:
            logger.warning("Spline optimization did not converge: %s", res.message)

        T_opt = unpack(res.x)
        thrust_opt = self._spline_from_control_points(t_knots, T_opt)
        t_hist, x_hist = self.rocket.simulate(
            x0=x0,
            thrust_func=thrust_opt,
            t_final=t_final,
            dt=0.02,
        )

        # Sample thrust over trajectory for visualization.
        T_hist = np.zeros((t_hist.size, 3))
        for i, t in enumerate(t_hist):
            T_hist[i] = thrust_opt(t, x_hist[i])

        return {
            "t": t_hist,
            "x": x_hist,
            "T": T_hist,
            "t_knots": t_knots,
            "T_knots": T_opt,
            "opt_result": res,
        }

    # ------------------------------------------------------------------
    # Monte Carlo analysis
    # ------------------------------------------------------------------
    @profile_time
    def monte_carlo_analysis(
        self,
        n_trials: int = 1000,
        t_final: float = 40.0,
        thrust_profile: Callable[[float, Array], Array] | None = None,
        rng: np.random.Generator | None = None,
    ) -> Tuple[MonteCarloStats, Dict[str, Array]]:
        """Run Monte Carlo simulations under uncertainties and disturbances.

        The following uncertainties are modeled (all zero‑mean Gaussian):

        * Initial position: σ_x = σ_y = 5 m, σ_z = 20 m
        * Initial velocity: σ_vx = σ_vy = 2 m/s, σ_vz = 5 m/s
        * Initial mass: σ_m = 30 kg
        * Wind disturbance: 2 m/s equivalent random horizontal force
        * Thrust noise: 5% multiplicative noise per time step

        Args:
            n_trials: Number of Monte Carlo trajectories.
            t_final: Final time horizon [s].
            thrust_profile: Nominal thrust profile; if None, a constant
                vertical thrust using bisection result is used.
            rng: Optional NumPy random generator for reproducibility.

        Returns:
            Tuple ``(stats, data)`` where ``stats`` contains aggregate
            measures and ``data`` contains arrays of terminal errors.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        cfg = self.rocket.config
        base_state = default_initial_state(cfg)

        # If no thrust profile specified, use bisection‑derived constant thrust.
        if thrust_profile is None:
            bis = self.constant_thrust_bisection(t_final=t_final)
            T_vec = np.array([0.0, 0.0, bis.T_opt], dtype=float)
            thrust_profile = constant_thrust_profile(T_vec)

        # Pre‑allocate result arrays.
        pos_errors = np.zeros(n_trials)
        vel_errors = np.zeros(n_trials)
        success_flags = np.zeros(n_trials, dtype=bool)

        # Disturbance standard deviations
        sigma_pos = np.array([5.0, 5.0, 20.0])
        sigma_vel = np.array([2.0, 2.0, 5.0])
        sigma_m = 30.0

        wind_sigma = 2.0  # m/s equivalent
        thrust_noise_std = 0.05  # 5% multiplicative noise

        def disturbed_thrust(t: float, x: Array) -> Array:
            """Thrust with multiplicative noise and simple wind disturbance."""
            T_nom = thrust_profile(t, x)
            # Apply thrust magnitude noise.
            noise = rng.normal(0.0, thrust_noise_std)
            T_noisy = (1.0 + noise) * T_nom

            # Simple wind: random horizontal acceleration proxy.
            wind_dir = rng.normal(0.0, 1.0, size=3)
            wind_dir[2] = 0.0  # horizontal only
            norm = np.linalg.norm(wind_dir)
            if norm > 1e-8:
                wind_dir /= norm
            wind_mag = rng.normal(0.0, wind_sigma)
            wind_force = cfg.m0 * wind_mag * wind_dir  # crude approximation

            return T_noisy + wind_force

        for i in range(n_trials):
            # Perturb initial state.
            dx0 = rng.normal(0.0, sigma_pos)
            dv0 = rng.normal(0.0, sigma_vel)
            dm0 = rng.normal(0.0, sigma_m)

            x0 = base_state.copy()
            x0[0:3] += dx0
            x0[3:6] += dv0
            x0[6] = max(cfg.m_dry, base_state[6] + dm0)

            t_hist, x_hist = self.rocket.simulate(
                x0=x0,
                thrust_func=disturbed_thrust,
                t_final=t_final,
                dt=0.05,
            )
            xf = x_hist[-1]
            r_f = xf[0:3]
            v_f = xf[3:6]

            pos_err = float(np.linalg.norm(r_f))
            vel_err = float(np.linalg.norm(v_f))

            pos_errors[i] = pos_err
            vel_errors[i] = vel_err

            success_flags[i] = (pos_err < 5.0) and (vel_err < 2.0)

            if (i + 1) % max(1, n_trials // 10) == 0:
                logger.info(
                    "Monte Carlo progress: %d/%d trials complete.", i + 1, n_trials
                )

        success_rate = float(np.mean(success_flags))
        mean_pos_error = float(np.mean(pos_errors))
        p95_pos_error = float(np.percentile(pos_errors, 95.0))
        mean_vel_error = float(np.mean(vel_errors))
        p95_vel_error = float(np.percentile(vel_errors, 95.0))

        stats = MonteCarloStats(
            success_rate=success_rate,
            mean_pos_error=mean_pos_error,
            p95_pos_error=p95_pos_error,
            mean_vel_error=mean_vel_error,
            p95_vel_error=p95_vel_error,
        )

        data = {
            "pos_errors": pos_errors,
            "vel_errors": vel_errors,
            "success_flags": success_flags,
        }
        return stats, data


__all__ = [
    "ClassicalSolver",
    "BisectionResult",
    "MonteCarloStats",
]


