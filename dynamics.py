"""Rocket dynamics and numerical integration utilities for 3D powered descent.

This module implements a 3D rocket dynamics model and high‑accuracy
Runge–Kutta integration suitable for powered descent guidance studies.
The continuous‑time model and mass depletion follow the formulations in
Meditch (1964) and Açıkmeşe & Ploen (2007), while the implementation is
tailored for numerical optimization and Monte Carlo analysis.

References:
    Meditch, J.S., "On the Problem of Optimal Thrust Programming for a
        Lunar Soft Landing", IEEE Trans. Automatic Control, 1964.
    Açıkmeşe, B., and Ploen, S., "Convex Programming Approach to Powered
        Descent Guidance for Mars Landing", Journal of Guidance, Control,
        and Dynamics, 2007.
    Blackmore, L., Açıkmeşe, B., and Scharf, D.P., "Minimum-Effort
        Guidance for Planetary Landing Using Convex Optimization",
        Journal of Guidance, Control, and Dynamics, 2010.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

logger = logging.getLogger(__name__)


Array = np.ndarray


@dataclass
class RocketConfig:
    """Configuration parameters for the 3D rocket model.

    All units are SI (m, s, kg, N).
    """

    m0: float = 1000.0  # initial mass [kg]
    m_dry: float = 500.0  # dry mass [kg]
    T_min: float = 6000.0  # minimum thrust magnitude [N]
    T_max: float = 15000.0  # maximum thrust magnitude [N]
    Isp: float = 300.0  # specific impulse [s]
    g0: float = 9.81  # standard gravity [m/s^2]
    g: float = 9.81  # local gravity [m/s^2]
    Cd: float = 0.5  # drag coefficient [-]
    A: float = 10.0  # reference area [m^2]
    rho0: float = 1.225  # sea‑level air density [kg/m^3]
    H: float = 8500.0  # scale height for exponential atmosphere [m]
    thrust_gimbal_deg: float = 30.0  # maximum gimbal angle from vertical [deg]

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is non‑physical.
        """
        if self.m_dry <= 0 or self.m0 <= 0:
            raise ValueError("Masses must be positive.")
        if self.m_dry >= self.m0:
            raise ValueError("Dry mass must be less than initial mass.")
        if self.T_min <= 0 or self.T_max <= 0:
            raise ValueError("Thrust bounds must be positive.")
        if self.T_min >= self.T_max:
            raise ValueError("T_min must be strictly less than T_max.")
        if self.Isp <= 0:
            raise ValueError("Specific impulse must be positive.")
        if self.A <= 0:
            raise ValueError("Reference area must be positive.")
        if self.rho0 <= 0 or self.H <= 0:
            raise ValueError("Atmosphere parameters must be positive.")


class Rocket3D:
    """Three‑dimensional rocket dynamics and RK4 integrator.

    The state vector is x = [x, y, z, vx, vy, vz, m]^T. The control input
    is the thrust vector T = [Tx, Ty, Tz]^T in Newtons, expressed in an
    inertial frame with +z pointing upward. The equations include gravity,
    atmospheric drag, and mass depletion due to thrust:

        dx/dt = vx
        dy/dt = vy
        dz/dt = vz
        dvx/dt = (Tx + Dx) / m
        dvy/dt = (Ty + Dy) / m
        dvz/dt = (Tz + Dz) / m - g
        dm/dt = -||T|| / (Isp * g0)

    where D is a quadratic drag force proportional to atmospheric density.
    """

    def __init__(self, config: RocketConfig | None = None) -> None:
        self.config = config or RocketConfig()
        self.config.validate()

    # ------------------------------------------------------------------
    # Core physics
    # ------------------------------------------------------------------
    def atmospheric_density(self, z: float) -> float:
        """Compute atmospheric density at altitude z.

        Args:
            z: Altitude above the landing pad [m]. Values below zero are clamped.

        Returns:
            Air density at altitude [kg/m^3].
        """
        z_eff = max(0.0, float(z))
        rho = self.config.rho0 * np.exp(-z_eff / self.config.H)
        return float(rho)

    def drag_force(self, v: Array, z: float) -> Array:
        """Compute aerodynamic drag force vector.

        Args:
            v: Velocity vector [vx, vy, vz] [m/s].
            z: Altitude [m].

        Returns:
            Drag force vector [Dx, Dy, Dz] [N].
        """
        speed = float(np.linalg.norm(v))
        if speed == 0.0:
            return np.zeros(3)
        rho = self.atmospheric_density(z)
        # Quadratic drag magnitude: 0.5 * rho * Cd * A * v^2
        drag_mag = 0.5 * rho * self.config.Cd * self.config.A * speed ** 2
        drag_vec = -drag_mag * (v / speed)
        return drag_vec

    def dynamics(self, t: float, x: Array, thrust: Array) -> Array:
        """Compute time derivative of the state.

        Args:
            t: Time [s]. (Unused but included for standard ODE interface.)
            x: State vector [x, y, z, vx, vy, vz, m].
            thrust: Thrust vector [Tx, Ty, Tz] [N].

        Returns:
            State derivative dx/dt as a (7,) array.

        Raises:
            ValueError: If mass is non‑positive or NaN values are detected.
        """
        if np.any(~np.isfinite(x)):
            raise ValueError("Non‑finite state encountered in dynamics.")
        if np.any(~np.isfinite(thrust)):
            raise ValueError("Non‑finite thrust encountered in dynamics.")

        pos = x[0:3]
        vel = x[3:6]
        m = float(x[6])

        if m <= 0.0:
            raise ValueError("Mass must remain positive in dynamics.")

        # Enforce simple thrust bounds for safety (soft clipping).
        T = np.asarray(thrust, dtype=float).reshape(3)
        T_norm = float(np.linalg.norm(T))
        if T_norm > 0:
            T_clamped = np.clip(T_norm, self.config.T_min, self.config.T_max) * (
                T / T_norm
            )
        else:
            T_clamped = np.zeros(3)

        # Optional gimbal limit: enforce angle between thrust and +z axis.
        if T_norm > 0:
            ez = np.array([0.0, 0.0, 1.0])
            cos_theta = float(np.dot(T_clamped, ez) / np.linalg.norm(T_clamped))
            cos_min = np.cos(np.deg2rad(self.config.thrust_gimbal_deg))
            if cos_theta < cos_min:
                # Project onto cone surface while preserving magnitude.
                # This is a simple geometric correction, not a physical controller.
                lateral = T_clamped.copy()
                lateral[2] = 0.0
                lat_norm = float(np.linalg.norm(lateral))
                if lat_norm > 1e-8:
                    # Set vertical component to satisfy cos(theta)=cos_min.
                    T_mag = float(np.linalg.norm(T_clamped))
                    Tz = T_mag * cos_min
                    Txy = np.sqrt(max(T_mag ** 2 - Tz ** 2, 0.0))
                    lateral_dir = lateral / lat_norm
                    T_clamped = Txy * lateral_dir
                    T_clamped[2] = Tz

        drag = self.drag_force(vel, pos[2])

        acc = (T_clamped + drag) / m
        acc[2] -= self.config.g

        # Mass depletion from thrust magnitude.
        T_used = float(np.linalg.norm(T_clamped))
        mdot = -T_used / (self.config.Isp * self.config.g0)

        dxdt = np.zeros(7)
        dxdt[0:3] = vel
        dxdt[3:6] = acc
        dxdt[6] = mdot
        return dxdt

    # ------------------------------------------------------------------
    # RK4 integration
    # ------------------------------------------------------------------
    def rk4_step(
        self,
        t: float,
        x: Array,
        dt: float,
        thrust_func: Callable[[float, Array], Array],
    ) -> Array:
        """Perform a single fourth‑order Runge–Kutta integration step.

        Args:
            t: Current time [s].
            x: Current state vector shape (7,).
            dt: Time step [s].
            thrust_func: Callable ``T = thrust_func(t, x)``.

        Returns:
            State vector at time t + dt.

        Raises:
            ValueError: If resulting mass drops below dry mass.
        """
        if dt <= 0.0:
            raise ValueError("Time step dt must be positive.")

        k1 = self.dynamics(t, x, thrust_func(t, x))
        k2 = self.dynamics(t + 0.5 * dt, x + 0.5 * dt * k1, thrust_func(t + 0.5 * dt, x + 0.5 * dt * k1))
        k3 = self.dynamics(t + 0.5 * dt, x + 0.5 * dt * k2, thrust_func(t + 0.5 * dt, x + 0.5 * dt * k2))
        k4 = self.dynamics(t + dt, x + dt * k3, thrust_func(t + dt, x + dt * k3))

        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if x_next[6] < self.config.m_dry - 1e-6:
            raise ValueError(
                f"Mass dropped below dry mass ({x_next[6]:.2f} kg < {self.config.m_dry:.2f} kg)."
            )
        return x_next

    def simulate(
        self,
        x0: Array,
        thrust_func: Callable[[float, Array], Array],
        t_final: float,
        dt: float = 0.02,
    ) -> Tuple[Array, Array]:
        """Simulate the rocket trajectory using fixed‑step RK4.

        Args:
            x0: Initial state vector [x, y, z, vx, vy, vz, m].
            thrust_func: Callable mapping (t, x) -> 3D thrust vector [N].
            t_final: Final simulation time [s].
            dt: Time step [s]. Defaults to 0.02 (50 Hz).

        Returns:
            Tuple ``(t_hist, x_hist)`` where:
                t_hist: Array of shape (N,) of time stamps.
                x_hist: Array of shape (N, 7) of state history.

        Raises:
            ValueError: If initial mass < dry mass or invalid inputs.

        Notes:
            For energy validation, see :meth:`compute_energy`.
        """
        x0 = np.asarray(x0, dtype=float).reshape(7)
        if x0[6] < self.config.m_dry:
            raise ValueError(
                f"Initial mass {x0[6]:.2f} kg is below dry mass {self.config.m_dry:.2f} kg."
            )
        if t_final <= 0.0:
            raise ValueError("t_final must be positive.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        n_steps = int(np.ceil(t_final / dt)) + 1
        t_hist = np.linspace(0.0, t_final, n_steps)
        x_hist = np.zeros((n_steps, 7))
        x_hist[0] = x0

        t = 0.0
        x = x0.copy()
        for k in range(1, n_steps):
            try:
                x = self.rk4_step(t, x, dt, thrust_func)
            except ValueError as exc:
                logger.error("Simulation aborted at step %d, t=%.3f: %s", k, t, exc)
                # Truncate history and return what we have so far.
                t_hist = t_hist[:k]
                x_hist = x_hist[:k]
                break
            x_hist[k] = x
            t += dt

        return t_hist, x_hist

    # ------------------------------------------------------------------
    # Energy accounting
    # ------------------------------------------------------------------
    def compute_energy(self, x: Array) -> Tuple[float, float]:
        """Compute mechanical energy components for a given state.

        Args:
            x: State vector [x, y, z, vx, vy, vz, m].

        Returns:
            Tuple ``(ke, pe)`` where:
                ke: Kinetic energy [J].
                pe: Gravitational potential energy [J] relative to z=0.
        """
        v = x[3:6]
        z = float(x[2])
        m = float(x[6])

        ke = 0.5 * m * float(np.dot(v, v))
        pe = m * self.config.g * z
        return float(ke), float(pe)

    def energy_history(self, x_hist: Array) -> Tuple[Array, Array, Array]:
        """Compute kinetic, potential, and total energy time histories.

        Args:
            x_hist: State history of shape (N, 7).

        Returns:
            Tuple ``(ke, pe, etot)`` each of shape (N,).
        """
        n = x_hist.shape[0]
        ke = np.zeros(n)
        pe = np.zeros(n)
        for i in range(n):
            ke[i], pe[i] = self.compute_energy(x_hist[i])
        etot = ke + pe
        return ke, pe, etot


def default_initial_state(config: RocketConfig | None = None) -> Array:
    """Return the default initial state vector for final descent.

    The default corresponds to a vehicle at 1 km altitude directly above
    the pad, with a small cross‑range velocity and descending:

        position: [0, 0, 1000] m
        velocity: [5, 0, -50] m/s
        mass: 1000 kg

    Args:
        config: Optional :class:`RocketConfig` to obtain ``m0``. If not
            provided, a default configuration is used.

    Returns:
        State vector [x, y, z, vx, vy, vz, m].
    """
    cfg = config or RocketConfig()
    return np.array([0.0, 0.0, 1000.0, 5.0, 0.0, -50.0, cfg.m0], dtype=float)


def constant_thrust_profile(T_vec: Array) -> Callable[[float, Array], Array]:
    """Create a constant‑thrust function suitable for :meth:`Rocket3D.simulate`.

    Args:
        T_vec: Constant thrust vector [Tx, Ty, Tz] [N].

    Returns:
        Callable mapping (t, x) -> T_vec.
    """

    T = np.asarray(T_vec, dtype=float).reshape(3)

    def _thrust(_t: float, _x: Array) -> Array:  # pragma: no cover - trivial
        return T

    return _thrust


__all__ = [
    "RocketConfig",
    "Rocket3D",
    "default_initial_state",
    "constant_thrust_profile",
]


