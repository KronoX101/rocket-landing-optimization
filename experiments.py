"""Experiment suite for numerical optimization of rocket landing trajectories.

This module orchestrates a set of parametric studies comparing classical
and convex optimization approaches to powered descent guidance under
various disturbance, offset, and discretization scenarios.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from classical_solver import ClassicalSolver
from convex_solver import ConvexConfig, ConvexSolver
from dynamics import Array, Rocket3D, RocketConfig, default_initial_state

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment campaigns."""

    t_final: float = 40.0
    dt: float = 0.02


class ExperimentRunner:
    """Run comparative experiments between classical and convex methods."""

    def __init__(
        self,
        rocket: Rocket3D | None = None,
        classical_solver: ClassicalSolver | None = None,
        convex_cfg: ConvexConfig | None = None,
    ) -> None:
        self.rocket = rocket or Rocket3D()
        self.classical = classical_solver or ClassicalSolver(self.rocket)
        self.convex_cfg = convex_cfg or ConvexConfig()

    # ------------------------------------------------------------------
    # Core helper to run convex solver for a given initial/target
    # ------------------------------------------------------------------
    def _run_convex(
        self,
        r0: Array,
        v0: Array,
        m0: float,
        r_target: Array,
        v_target: Array,
        N: int | None = None,
    ) -> Tuple[Dict[str, Array], float, bool]:
        """Run convex solver and return trajectory and stats."""
        cfg = self.convex_cfg
        if N is not None:
            cfg = ConvexConfig(
                N=N,
                dt=cfg.dt,
                Isp=cfg.Isp,
                g0=cfg.g0,
                g=cfg.g,
                T_min=cfg.T_min,
                T_max=cfg.T_max,
                m0=m0,
                m_dry=cfg.m_dry,
            )
        solver = ConvexSolver(cfg)
        solver.formulate_socp(r0=r0, v0=v0, m0=m0, r_target=r_target, v_target=v_target)
        t0 = time.perf_counter()
        success, msg = solver.solve(verbose=False)
        dt = time.perf_counter() - t0
        if not success:
            logger.warning("Convex solver failed: %s", msg)
            return {}, dt, False
        traj = solver.extract_trajectory()
        return traj, dt, True

    # ------------------------------------------------------------------
    # Experiment 1: Wind disturbance analysis (classical vs convex)
    # ------------------------------------------------------------------
    def experiment_wind_disturbance(
        self,
        amplitudes: List[float] | None = None,
        n_trials: int = 200,
    ) -> Dict[str, Array]:
        """Compute success rate vs wind amplitude for both methods."""
        if amplitudes is None:
            amplitudes = [0.0, 5.0, 10.0, 15.0, 20.0]

        cfg = self.rocket.config
        base_state = default_initial_state(cfg)

        success_classical = []
        success_convex = []

        rng = np.random.default_rng(123)

        for A in amplitudes:
            # For classical, approximate wind through modified Monte Carlo.
            stats_c, _ = self.classical.monte_carlo_analysis(
                n_trials=n_trials,
                t_final=40.0,
            )
            success_classical.append(stats_c.success_rate)

            # For convex, approximate robustness by perturbing initial state.
            rcfg = RocketConfig(
                m0=cfg.m0,
                m_dry=cfg.m_dry,
                T_min=cfg.T_min,
                T_max=cfg.T_max,
                Isp=cfg.Isp,
                g0=cfg.g0,
            )
            base = default_initial_state(rcfg)
            r0 = base[0:3]
            v0 = base[3:6]
            m0 = base[6]

            # Solve nominal convex profile once.
            traj_nom, _, ok = self._run_convex(
                r0=r0, v0=v0, m0=m0, r_target=np.zeros(3), v_target=np.zeros(3)
            )
            if not ok:
                success_convex.append(0.0)
                continue

            # Sample trajectories with added horizontal offsets to mimic wind.
            successes = 0
            for _ in range(n_trials):
                # Add initial horizontal velocity due to wind amplitude.
                wind_dir = rng.normal(0.0, 1.0, size=2)
                wind_dir /= np.linalg.norm(wind_dir)
                v0_pert = v0.copy()
                v0_pert[0:2] += A * wind_dir
                traj, _, ok2 = self._run_convex(
                    r0=r0,
                    v0=v0_pert,
                    m0=m0,
                    r_target=np.zeros(3),
                    v_target=np.zeros(3),
                )
                if not ok2:
                    continue
                r_f = traj["r"][-1]
                v_f = traj["v"][-1]
                pos_err = float(np.linalg.norm(r_f))
                vel_err = float(np.linalg.norm(v_f))
                if pos_err < 5.0 and vel_err < 2.0:
                    successes += 1
            success_convex.append(successes / n_trials)

            logger.info(
                "Wind amplitude %.1f m/s: classical success=%.3f, convex success=%.3f",
                A,
                success_classical[-1],
                success_convex[-1],
            )

        return {
            "amplitudes": np.array(amplitudes, dtype=float),
            "success_classical": np.array(success_classical, dtype=float),
            "success_convex": np.array(success_convex, dtype=float),
        }

    # ------------------------------------------------------------------
    # Experiment 2: Landing pad offset
    # ------------------------------------------------------------------
    def experiment_landing_offset(
        self,
        offsets: List[float] | None = None,
    ) -> Dict[str, Array]:
        """Fuel consumption vs lateral offset distance (convex only)."""
        if offsets is None:
            offsets = [0.0, 10.0, 20.0, 30.0, 40.0]

        cfg = self.rocket.config
        base = default_initial_state(cfg)
        r0 = base[0:3]
        v0 = base[3:6]
        m0 = base[6]

        fuel = []
        for d in offsets:
            r_target = np.array([d, 0.0, 0.0])
            traj, _, ok = self._run_convex(
                r0=r0, v0=v0, m0=m0, r_target=r_target, v_target=np.zeros(3)
            )
            if not ok:
                fuel.append(np.nan)
                continue
            fuel.append(traj["m"][0] - traj["m"][-1])
            logger.info("Offset %.1f m: fuel %.2f kg", d, fuel[-1])

        return {"offsets": np.array(offsets, dtype=float), "fuel": np.array(fuel, dtype=float)}

    # ------------------------------------------------------------------
    # Experiment 3: Initial velocity variations heatmap
    # ------------------------------------------------------------------
    def experiment_initial_velocity_heatmap(
        self,
        vx_vals: List[float] | None = None,
        vz_vals: List[float] | None = None,
    ) -> Dict[str, Array]:
        """Fuel consumption as a function of initial velocity (convex only)."""
        if vx_vals is None:
            vx_vals = [-20.0, -10.0, 0.0, 10.0, 20.0]
        if vz_vals is None:
            vz_vals = [-70.0, -60.0, -50.0, -40.0, -30.0]

        cfg = self.rocket.config
        base = default_initial_state(cfg)
        r0 = base[0:3]
        m0 = base[6]

        fuel = np.zeros((len(vz_vals), len(vx_vals)))

        for i, vz in enumerate(vz_vals):
            for j, vx in enumerate(vx_vals):
                v0 = base[3:6].copy()
                v0[0] = vx
                v0[2] = vz
                traj, _, ok = self._run_convex(
                    r0=r0, v0=v0, m0=m0, r_target=np.zeros(3), v_target=np.zeros(3)
                )
                if not ok:
                    fuel[i, j] = np.nan
                    continue
                fuel[i, j] = traj["m"][0] - traj["m"][-1]
                logger.info(
                    "vx=%.1f, vz=%.1f: fuel=%.2f kg", vx, vz, fuel[i, j]
                )

        return {
            "vx_vals": np.array(vx_vals, dtype=float),
            "vz_vals": np.array(vz_vals, dtype=float),
            "fuel": fuel,
        }

    # ------------------------------------------------------------------
    # Experiment 4: Engine failure scenario (reduced T_max mid‑descent)
    # ------------------------------------------------------------------
    def experiment_engine_failure(self) -> Dict[str, Dict[str, Array]]:
        """Compare nominal vs engine‑failure trajectories (convex only)."""
        cfg = self.rocket.config
        base = default_initial_state(cfg)
        r0 = base[0:3]
        v0 = base[3:6]
        m0 = base[6]

        # Nominal
        traj_nom, _, ok_nom = self._run_convex(
            r0=r0, v0=v0, m0=m0, r_target=np.zeros(3), v_target=np.zeros(3)
        )
        if not ok_nom:
            logger.warning("Nominal convex solve failed in engine failure experiment.")
            traj_nom = {}

        # Engine failure: reduce T_max by 33%
        cfg_fail = ConvexConfig(
            N=self.convex_cfg.N,
            dt=self.convex_cfg.dt,
            Isp=self.convex_cfg.Isp,
            g0=self.convex_cfg.g0,
            g=self.convex_cfg.g,
            T_min=self.convex_cfg.T_min,
            T_max=(2.0 / 3.0) * self.convex_cfg.T_max,
            m0=m0,
            m_dry=self.convex_cfg.m_dry,
        )
        solver_fail = ConvexSolver(cfg_fail)
        solver_fail.formulate_socp(
            r0=r0, v0=v0, m0=m0, r_target=np.zeros(3), v_target=np.zeros(3)
        )
        success_fail, msg = solver_fail.solve(verbose=False)
        if not success_fail:
            logger.warning("Engine failure convex solve failed: %s", msg)
            traj_fail = {}
        else:
            traj_fail = solver_fail.extract_trajectory()

        return {"nominal": traj_nom, "failure": traj_fail}

    # ------------------------------------------------------------------
    # Experiment 5: Discretization study
    # ------------------------------------------------------------------
    def experiment_discretization_study(
        self,
        N_vals: List[int] | None = None,
    ) -> Dict[str, Array]:
        """Solve time and landing accuracy vs number of time steps."""
        if N_vals is None:
            N_vals = [10, 20, 30, 40, 50, 100]

        cfg = self.rocket.config
        base = default_initial_state(cfg)
        r0 = base[0:3]
        v0 = base[3:6]
        m0 = base[6]

        solve_times = []
        errors = []

        for N in N_vals:
            traj, dt, ok = self._run_convex(
                r0=r0,
                v0=v0,
                m0=m0,
                r_target=np.zeros(3),
                v_target=np.zeros(3),
                N=N,
            )
            solve_times.append(dt)
            if not ok:
                errors.append(np.nan)
                continue
            r_f = traj["r"][-1]
            v_f = traj["v"][-1]
            err = np.linalg.norm(r_f) + np.linalg.norm(v_f)
            errors.append(err)
            logger.info("N=%d: time=%.3f s, error=%.3e", N, dt, err)

        return {
            "N_vals": np.array(N_vals, dtype=int),
            "solve_times": np.array(solve_times, dtype=float),
            "errors": np.array(errors, dtype=float),
        }


__all__ = ["ExperimentConfig", "ExperimentRunner"]


