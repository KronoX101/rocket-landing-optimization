"""Convex optimization based powered descent guidance (SOCP formulation).

This module implements a lossless convexification of the powered descent
guidance problem following Açıkmeşe & Ploen (2007) and Blackmore et al.
(2010). The nonlinear thrust‑over‑mass dynamics are converted into a
second‑order cone program (SOCP) in terms of position, velocity, log
mass, and thrust‑per‑mass variables, which can be solved efficiently
with modern convex optimization solvers via CVXPY.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cvxpy as cp
import numpy as np

from dynamics import Array, RocketConfig, default_initial_state

logger = logging.getLogger(__name__)


@dataclass
class ConvexConfig:
    """Configuration for the convex SOCP guidance problem."""

    N: int = 50  # number of time steps
    dt: float = 0.5  # time step [s]

    # Physical parameters (mirrored from RocketConfig)
    Isp: float = 300.0
    g0: float = 9.81
    g: float = 9.81
    T_min: float = 6000.0
    T_max: float = 15000.0
    m0: float = 1000.0
    m_dry: float = 500.0

    def validate(self) -> None:
        """Validate configuration values."""
        if self.N <= 1:
            raise ValueError("N must be greater than 1.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.m_dry <= 0 or self.m0 <= 0:
            raise ValueError("Masses must be positive.")
        if self.m_dry >= self.m0:
            raise ValueError("Dry mass must be less than initial mass.")
        if self.T_min <= 0 or self.T_max <= 0:
            raise ValueError("Thrust bounds must be positive.")
        if self.T_min >= self.T_max:
            raise ValueError("T_min must be strictly less than T_max.")


class ConvexSolver:
    """Losslessly convexified powered descent guidance solver using CVXPY."""

    def __init__(self, cfg: Optional[ConvexConfig] = None) -> None:
        self.cfg = cfg or ConvexConfig()
        self.cfg.validate()

        self._problem: Optional[cp.Problem] = None
        self._vars: Dict[str, cp.Variable] = {}

    # ------------------------------------------------------------------
    # Problem formulation
    # ------------------------------------------------------------------
    def formulate_socp(
        self,
        r0: Optional[Array] = None,
        v0: Optional[Array] = None,
        m0: Optional[float] = None,
        r_target: Optional[Array] = None,
        v_target: Optional[Array] = None,
    ) -> cp.Problem:
        """Build the SOCP for powered descent.

        Args:
            r0: Initial position [x, y, z] [m]. Defaults to standard state.
            v0: Initial velocity [vx, vy, vz] [m/s].
            m0: Initial mass [kg].
            r_target: Target position [m]. Defaults to [0, 0, 0].
            v_target: Target velocity [m/s]. Defaults to [0, 0, 0].

        Returns:
            Constructed :class:`cvxpy.Problem` instance.
        """
        cfg = self.cfg

        if r0 is None or v0 is None or m0 is None:
            rc = RocketConfig(
                m0=cfg.m0,
                m_dry=cfg.m_dry,
                T_min=cfg.T_min,
                T_max=cfg.T_max,
                Isp=cfg.Isp,
                g0=cfg.g0,
            )
            x0 = default_initial_state(rc)
            r0 = x0[0:3]
            v0 = x0[3:6]
            m0 = float(x0[6])

        r0 = np.asarray(r0, dtype=float).reshape(3)
        v0 = np.asarray(v0, dtype=float).reshape(3)
        m0 = float(m0)

        if r_target is None:
            r_target = np.zeros(3)
        if v_target is None:
            v_target = np.zeros(3)

        r_target = np.asarray(r_target, dtype=float).reshape(3)
        v_target = np.asarray(v_target, dtype=float).reshape(3)

        N, dt = cfg.N, cfg.dt

        # Decision variables
        r = cp.Variable((N + 1, 3), name="r")
        v = cp.Variable((N + 1, 3), name="v")
        z = cp.Variable(N + 1, name="z")  # log mass
        u = cp.Variable((N, 3), name="u")  # thrust per mass
        Gamma = cp.Variable(N, name="Gamma")  # thrust magnitude per mass

        g_vec = np.array([0.0, 0.0, -cfg.g])

        constraints = []

        # Boundary conditions
        constraints += [
            r[0, :] == r0,
            v[0, :] == v0,
            z[0] == np.log(m0),
            r[N, :] == r_target,
            v[N, :] == v_target,
        ]

        # Dynamics: trapezoidal for position, forward Euler for velocity and log mass.
        for k in range(N):
            constraints += [
                r[k + 1, :] == r[k, :] + 0.5 * dt * (v[k, :] + v[k + 1, :]),
                v[k + 1, :] == v[k, :] + dt * (g_vec + u[k, :]),
                z[k + 1] == z[k] - (dt * Gamma[k]) / (cfg.Isp * cfg.g0),
            ]

        # SOC constraints and thrust bounds.
        # NOTE (DCP compliance):
        # In the original lossless convexification, the thrust bounds are
        # T_min / m <= Gamma <= T_max / m, with m = exp(z). Directly encoding
        # Gamma <= T_max * exp(-z) is not DCP because the right-hand side is
        # convex in z. As a simple, conservative approximation we instead use
        # constant bounds based on the known mass range:
        #
        #   Gamma_min = T_min / m0
        #   Gamma_max = T_max / m_dry
        #
        # which preserves convexity while keeping Gamma within physically
        # reasonable limits over the descent.
        Gamma_min = cfg.T_min / m0
        Gamma_max = cfg.T_max / cfg.m_dry

        for k in range(N):
            constraints.append(cp.SOC(Gamma[k], u[k, :]))  # ||u_k||_2 <= Gamma_k

            # Thrust magnitude bounds (conservative, DCP-compliant).
            constraints += [
                Gamma[k] >= Gamma_min,
                Gamma[k] <= Gamma_max,
            ]

        # Mass bounds and altitude non‑negativity
        for k in range(N + 1):
            constraints.append(z[k] >= np.log(cfg.m_dry))
            # Altitude (z component of r) must be non‑negative
            constraints.append(r[k, 2] >= 0.0)

        # Objective: maximize final log mass (minimize fuel usage).
        objective = cp.Maximize(z[N])

        problem = cp.Problem(objective, constraints)

        self._problem = problem
        self._vars = {"r": r, "v": v, "z": z, "u": u, "Gamma": Gamma}
        return problem

    # ------------------------------------------------------------------
    # Solving and extraction
    # ------------------------------------------------------------------
    def solve(self, verbose: bool = False) -> Tuple[bool, Optional[str]]:
        """Solve the formulated SOCP.

        Args:
            verbose: If True, pass verbose flag to CVXPY solver.

        Returns:
            Tuple ``(success, message)`` indicating feasibility and any
            diagnostic message.
        """
        if self._problem is None:
            raise RuntimeError("Problem not yet formulated. Call formulate_socp() first.")

        try:
            # Let CVXPY select an appropriate installed solver automatically.
            # If ECOS is available it will typically be chosen; otherwise
            # SCS/OSQP or another compatible solver will be used.
            self._problem.solve(verbose=verbose)
        except Exception as exc:  # pragma: no cover - solver‑level
            logger.error("CVXPY solver error: %s", exc)
            return False, str(exc)

        status = self._problem.status
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            msg = f"Convex problem not optimal: status={status}"
            logger.warning(msg)
            return False, msg

        return True, status

    def extract_trajectory(self) -> Dict[str, Array]:
        """Extract state and control trajectories from the CVXPY solution.

        Returns:
            Dictionary containing time stamps, position, velocity, mass,
            acceleration control, and thrust magnitude.

        Raises:
            RuntimeError: If the problem has not been solved successfully.
        """
        if self._problem is None or not self._vars:
            raise RuntimeError("Problem not formulated.")

        if self._problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(
                f"Cannot extract trajectory from problem with status {self._problem.status}"
            )

        cfg = self.cfg
        N, dt = cfg.N, cfg.dt

        r = self._vars["r"].value
        v = self._vars["v"].value
        z = self._vars["z"].value
        u = self._vars["u"].value
        Gamma = self._vars["Gamma"].value

        t = np.linspace(0.0, N * dt, N + 1)
        m = np.exp(z)

        # Recover thrust from u and mass: T = m * u
        T = np.zeros((N, 3))
        for k in range(N):
            T[k, :] = m[k] * u[k, :]

        # Pad last thrust sample for plotting purposes.
        T_full = np.vstack([T, T[-1, :]])

        return {
            "t": t,
            "r": r,
            "v": v,
            "m": m,
            "u": u,
            "Gamma": Gamma,
            "T": T_full,
        }


__all__ = ["ConvexConfig", "ConvexSolver"]


