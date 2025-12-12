"""Classical numerical methods for powered descent guidance.

This module implements classical numerical techniques for landing a
reusable rocket stage in three dimensions:

* Bisection method to determine a constant thrust level that yields a
  soft vertical landing (root‑finding on terminal vertical velocity).
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
    z_final: float
    t_landing: float
    history: List[Tuple[int, float, float, float, float]]  # iter, Tz, vz_final, z_final, t_landing


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
    # Bisection for constant thrust - LANDING OPTIMIZED
    # ------------------------------------------------------------------
    @profile_time
    def constant_thrust_bisection(
        self,
        t_final: float = None,
        vz_tol: float = 0.5,
        z_tol: float = 1.0,
        max_iter: int = 50,
    ) -> BisectionResult:
        """Find a constant vertical thrust that yields a soft landing.

        The bisection search varies a constant thrust magnitude aligned
        with the +z axis and simulates until touchdown. The objective is
        to find thrust that results in landing at z≈0 with vz≈0.

        Args:
            t_final: Not used (kept for compatibility). Landing time is determined by simulation.
            vz_tol: Tolerance on landing vertical velocity [m/s]. Default 0.5 m/s.
            z_tol: Tolerance on landing altitude [m]. Default 1.0 m.
            max_iter: Maximum number of bisection iterations.

        Returns:
            :class:`BisectionResult` with optimal thrust and landing diagnostics.
        """
        cfg: RocketConfig = self.rocket.config
        T_low = cfg.T_min
        T_high = cfg.T_max

        history: List[Tuple[int, float, float, float, float]] = []

        def simulate_landing_with_Tz(Tz: float) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
            """Simulate until touchdown and return landing metrics."""
            T_vec = np.array([0.0, 0.0, Tz], dtype=float)
            x0 = default_initial_state(cfg)
            try:
                t_hist, x_hist = self.rocket.simulate_until_touchdown(
                    x0=x0,
                    thrust_func=constant_thrust_profile(T_vec),
                    dt=0.02,
                    altitude_tol=z_tol,
                )
                # Get final state
                if len(x_hist) > 0:
                    z_final = float(x_hist[-1, 2])
                    vz_final = float(x_hist[-1, 5])
                    t_landing = float(t_hist[-1])
                    return vz_final, z_final, t_landing, t_hist, x_hist
                else:
                    # Empty result - simulation failed
                    return 1000.0, 1000.0, 0.0, np.array([]), np.array([])
            except Exception as e:
                logger.debug("Simulation failed for Tz=%.2f: %s", Tz, e)
                # Return large penalty values to indicate failure
                return 1000.0, 1000.0, 0.0, np.array([]), np.array([])

        # Test initial bounds
        vz_low, z_low, t_low, _, _ = simulate_landing_with_Tz(T_low)
        vz_high, z_high, t_high, _, _ = simulate_landing_with_Tz(T_high)
        
        logger.info(
            "Bisection initial bracket: T_low=%.2f (z=%.2f, vz=%.2f), T_high=%.2f (z=%.2f, vz=%.2f)",
            T_low, z_low, vz_low, T_high, z_high, vz_high
        )

        converged = False
        T_opt = 0.5 * (T_low + T_high)
        vz_final = np.nan
        z_final = np.nan
        t_landing = np.nan
        min_iter = 10
        best_result = None
        best_error = float('inf')

        for i in range(max_iter):
            T_mid = 0.5 * (T_low + T_high)
            vz_mid, z_mid, t_mid, _, _ = simulate_landing_with_Tz(T_mid)
            
            # Compute landing quality (lower is better)
            landing_error = abs(z_mid) + 5.0 * abs(vz_mid)
            
            history.append((i, T_mid, vz_mid, z_mid, t_mid))
            logger.info(
                "Bisection iter %d: Tz=%.2f N, z_final=%.2f m, vz_final=%.3f m/s, t=%.2f s",
                i, T_mid, z_mid, vz_mid, t_mid
            )

            # Track best result
            if landing_error < best_error:
                best_error = landing_error
                best_result = (T_mid, vz_mid, z_mid, t_mid)

            # Check convergence: good landing if z≈0 and vz≈0
            if i >= min_iter and abs(z_mid) <= z_tol and abs(vz_mid) <= vz_tol:
                converged = True
                T_opt = T_mid
                vz_final = vz_mid
                z_final = z_mid
                t_landing = t_mid
                break

            # Bisection logic: optimize for landing velocity
            # If landing velocity is too negative (crashing): need more thrust
            # If landing velocity is positive (bouncing): need less thrust
            if vz_mid < -vz_tol:
                # Still crashing/descending too fast: need more thrust
                T_low = T_mid
                vz_low = vz_mid
                z_low = z_mid
            elif vz_mid > vz_tol:
                # Bouncing/ascending: need less thrust
                T_high = T_mid
                vz_high = vz_mid
                z_high = z_mid
            else:
                # Velocity is close to zero, optimize for altitude
                if z_mid < -z_tol:
                    # Went underground: need more thrust
                    T_low = T_mid
                    vz_low = vz_mid
                    z_low = z_mid
                elif z_mid > z_tol:
                    # Too high: this shouldn't happen if velocity is good, but if it does, need less thrust
                    T_high = T_mid
                    vz_high = vz_mid
                    z_high = z_mid
                else:
                    # Both are good!
                    converged = True
                    T_opt = T_mid
                    vz_final = vz_mid
                    z_final = z_mid
                    t_landing = t_mid
                    break

        else:
            # Use best result from search
            if best_result is not None:
                T_opt, vz_final, z_final, t_landing = best_result
                logger.info("Using best result: z=%.2f m, vz=%.3f m/s", z_final, vz_final)
            else:
                # Fallback to midpoint
                T_opt = 0.5 * (T_low + T_high)
                vz_final, z_final, t_landing, _, _ = simulate_landing_with_Tz(T_opt)

        return BisectionResult(
            T_opt=T_opt,
            n_iter=len(history),
            converged=converged,
            vz_final=vz_final,
            z_final=z_final,
            t_landing=t_landing,
            history=history,
        )
    
    def find_landing_trajectory(self, T_opt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate full landing trajectory with optimal thrust.
        
        Args:
            T_opt: Optimal thrust magnitude [N].
            
        Returns:
            Tuple (t_hist, x_hist) of landing trajectory.
        """
        T_vec = np.array([0.0, 0.0, T_opt], dtype=float)
        x0 = default_initial_state(self.rocket.config)
        t_hist, x_hist = self.rocket.simulate_until_touchdown(
            x0=x0,
            thrust_func=constant_thrust_profile(T_vec),
            dt=0.02,
            altitude_tol=0.5,
        )
        return t_hist, x_hist

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


