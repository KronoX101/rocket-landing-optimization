"""Visualization utilities for 3D rocket landing simulations.

This module provides publication‑quality plotting and animation
functions that compare classical numerical methods with convex
optimization based solutions for powered descent guidance.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from dynamics import Array

logger = logging.getLogger(__name__)


mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 120,
    }
)


@dataclass
class VisualConfig:
    """Configuration for plotting aesthetics."""

    cmap: str = "viridis"
    out_dir: str = "outputs"


class RocketVisualizer:
    """High‑level interface for simulation and optimization visualizations."""

    def __init__(self, cfg: Optional[VisualConfig] = None) -> None:
        self.cfg = cfg or VisualConfig()
        os.makedirs(self.cfg.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _save_figure(self, fig: plt.Figure, basename: str) -> None:
        """Save figure as both PNG and PDF."""
        png_path = os.path.join(self.cfg.out_dir, f"{basename}.png")
        pdf_path = os.path.join(self.cfg.out_dir, f"{basename}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        logger.info("Saved figure '%s' and '%s'", png_path, pdf_path)

    # ------------------------------------------------------------------
    # Core trajectory plots
    # ------------------------------------------------------------------
    def plot_3d_trajectory(
        self,
        t: Array,
        r: Array,
        T: Optional[Array] = None,
        label: str = "trajectory",
        basename: str = "trajectory_3d",
    ) -> None:
        """3D trajectory plot with optional thrust vectors."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(r[:, 0], r[:, 1], r[:, 2], label=label, color="C0")

        # Start and end markers
        ax.scatter(r[0, 0], r[0, 1], r[0, 2], color="green", s=60, label="Start")
        ax.scatter(r[-1, 0], r[-1, 1], r[-1, 2], color="red", s=60, label="End")

        # Landing pad as a circle on z=0.
        theta = np.linspace(0, 2 * np.pi, 100)
        pad_radius = 10.0
        pad_x = pad_radius * np.cos(theta)
        pad_y = pad_radius * np.sin(theta)
        pad_z = np.zeros_like(theta)
        ax.plot(pad_x, pad_y, pad_z, color="k", linestyle="--", label="Landing pad")

        # Optional thrust arrows
        if T is not None:
            idx = np.linspace(0, len(t) - 1, 20, dtype=int)
            scale = 0.001  # scale to visualize in position units
            ax.quiver(
                r[idx, 0],
                r[idx, 1],
                r[idx, 2],
                scale * T[idx, 0],
                scale * T[idx, 1],
                scale * T[idx, 2],
                length=1.0,
                color="C1",
                normalize=False,
                label="Thrust vectors",
            )

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title("3D Descent Trajectory")
        ax.legend(loc="best")
        ax.grid(True)

        # Equal aspect ratio
        max_range = np.array(
            [
                r[:, 0].max() - r[:, 0].min(),
                r[:, 1].max() - r[:, 1].min(),
                r[:, 2].max() - r[:, 2].min(),
            ]
        ).max()
        mid_x = 0.5 * (r[:, 0].max() + r[:, 0].min())
        mid_y = 0.5 * (r[:, 1].max() + r[:, 1].min())
        mid_z = 0.5 * (r[:, 2].max() + r[:, 2].min())
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(max(0.0, mid_z - max_range / 2), mid_z + max_range / 2)

        self._save_figure(fig, basename)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------
    def animate_descent(
        self,
        t: Array,
        r: Array,
        v: Array,
        m: Array,
        T: Optional[Array] = None,
        basename: str = "descent_animation",
        fps: int = 20,
        max_frames: int = 300,
        title: Optional[str] = None,
    ) -> None:
        """Generate an MP4 animation of the descent."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

        line, = ax.plot([], [], [], "C0-", lw=2)
        point = ax.scatter([], [], [], color="red", s=50)
        thrust_quiver = None

        # Text box for telemetry - positioned at upper left corner to avoid title overlap
        text = ax.text2D(
            0.01,
            0.99,
            "",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Pad outline
        theta = np.linspace(0, 2 * np.pi, 100)
        pad_radius = 10.0
        pad_x = pad_radius * np.cos(theta)
        pad_y = pad_radius * np.sin(theta)
        pad_z = np.zeros_like(theta)
        ax.plot(pad_x, pad_y, pad_z, color="k", linestyle="--")

        # Axes limits
        max_range = np.array(
            [
                r[:, 0].max() - r[:, 0].min(),
                r[:, 1].max() - r[:, 1].min(),
                r[:, 2].max() - r[:, 2].min(),
            ]
        ).max()
        mid_x = 0.5 * (r[:, 0].max() + r[:, 0].min())
        mid_y = 0.5 * (r[:, 1].max() + r[:, 1].min())
        mid_z = 0.5 * (r[:, 2].max() + r[:, 2].min())
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(0.0, mid_z + max_range / 2)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(title if title is not None else "3D Powered Descent Animation")

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            nonlocal thrust_quiver
            if thrust_quiver is not None:
                thrust_quiver.remove()
                thrust_quiver = None
            return line, point

        # To keep animations lightweight (especially for GIF export), we
        # optionally subsample the trajectory to at most ``max_frames``
        # frames.
        n_total = len(t)
        n_frames = min(n_total, max_frames)
        frame_indices = np.linspace(0, n_total - 1, n_frames, dtype=int)

        def update(frame_idx: int):
            frame = frame_indices[frame_idx]
            nonlocal thrust_quiver
            line.set_data(r[: frame + 1, 0], r[: frame + 1, 1])
            line.set_3d_properties(r[: frame + 1, 2])

            point._offsets3d = (
                np.array([r[frame, 0]]),
                np.array([r[frame, 1]]),
                np.array([r[frame, 2]]),
            )

            if thrust_quiver is not None:
                thrust_quiver.remove()
                thrust_quiver = None

            if T is not None:
                scale = 0.001
                thrust_quiver = ax.quiver(
                    [r[frame, 0]],
                    [r[frame, 1]],
                    [r[frame, 2]],
                    [scale * T[frame, 0]],
                    [scale * T[frame, 1]],
                    [scale * T[frame, 2]],
                    color="C1",
                )

            text.set_text(
                f"t = {t[frame]:5.1f} s\n"
                f"alt = {r[frame, 2]:7.2f} m\n"
                f"|v| = {np.linalg.norm(v[frame]):7.2f} m/s\n"
                f"m   = {m[frame]:7.2f} kg"
            )

            # Rotate view slowly
            azim = 45 + 0.3 * frame
            ax.view_init(elev=25, azim=azim)

            return line, point, text

        interval = 1000 * (t[-1] / n_frames) / (1.0 / fps)
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=n_frames,
            init_func=init,
            blit=False,
            interval=interval,
        )

        # Prefer ffmpeg for MP4 output. If ffmpeg is unavailable, skip
        # writing the animation to avoid very expensive GIF generation
        # via Pillow (which can be slow and memory‑intensive for 3D).
        mp4_path = os.path.join(self.cfg.out_dir, f"{basename}.mp4")
        if animation.writers.is_available("ffmpeg"):
            ani.save(mp4_path, writer="ffmpeg", fps=fps, dpi=200)
            logger.info("Saved MP4 animation '%s' using ffmpeg.", mp4_path)
        else:
            logger.warning(
                "ffmpeg unavailable; skipping animation '%s'. "
                "Install ffmpeg to enable MP4 export.",
                mp4_path,
            )
        plt.close(fig)

    def animate_bisection_touchdown(
        self,
        rocket,
        T_opt: float,
        basename: str = "bisection_touchdown_animation",
        fps: int = 20,
        max_frames: int = 500,
    ) -> None:
        """Generate an animation of the bisection solution running until touchdown.

        This method simulates the trajectory using the bisection-optimized thrust
        level and continues until the rocket touches down (z <= 0.1 m).

        Args:
            rocket: Rocket3D instance for simulation.
            T_opt: Optimal thrust magnitude from bisection [N] (vertical component).
            basename: Base filename for output animation.
            fps: Frames per second for animation.
            max_frames: Maximum number of animation frames.
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from dynamics import default_initial_state, constant_thrust_profile

        # Create constant vertical thrust profile
        T_vec = np.array([0.0, 0.0, T_opt], dtype=float)
        thrust_func = constant_thrust_profile(T_vec)

        # Get initial state
        x0 = default_initial_state(rocket.config)

        # Simulate until touchdown
        logger.info(
            "Simulating bisection trajectory until touchdown with T_opt=%.2f N",
            T_opt,
        )
        t, x = rocket.simulate_until_touchdown(x0, thrust_func, dt=0.02)

        r = x[:, 0:3]
        v = x[:, 3:6]
        m = x[:, 6]

        # Create constant thrust history for visualization
        T_hist = np.zeros((t.size, 3))
        T_hist[:] = T_vec[None, :]

        # Now create the animation
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

        line, = ax.plot([], [], [], "C0-", lw=2, label="Trajectory")
        point = ax.scatter([], [], [], color="red", s=50, label="Rocket")
        thrust_quiver = None

        # Text box for telemetry
        text = ax.text2D(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Pad outline
        theta = np.linspace(0, 2 * np.pi, 100)
        pad_radius = 10.0
        pad_x = pad_radius * np.cos(theta)
        pad_y = pad_radius * np.sin(theta)
        pad_z = np.zeros_like(theta)
        ax.plot(pad_x, pad_y, pad_z, color="k", linestyle="--", label="Landing pad")

        # Axes limits
        max_range = np.array(
            [
                r[:, 0].max() - r[:, 0].min(),
                r[:, 1].max() - r[:, 1].min(),
                r[:, 2].max() - r[:, 2].min(),
            ]
        ).max()
        mid_x = 0.5 * (r[:, 0].max() + r[:, 0].min())
        mid_y = 0.5 * (r[:, 1].max() + r[:, 1].min())
        mid_z = 0.5 * (r[:, 2].max() + r[:, 2].min())
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(0.0, mid_z + max_range / 2)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title("Bisection Landing")
        ax.legend(loc="upper right")

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            nonlocal thrust_quiver
            if thrust_quiver is not None:
                thrust_quiver.remove()
                thrust_quiver = None
            return line, point

        # Subsample for animation if needed
        n_total = len(t)
        n_frames = min(n_total, max_frames)
        frame_indices = np.linspace(0, n_total - 1, n_frames, dtype=int)

        def update(frame_idx: int):
            frame = frame_indices[frame_idx]
            nonlocal thrust_quiver
            line.set_data(r[: frame + 1, 0], r[: frame + 1, 1])
            line.set_3d_properties(r[: frame + 1, 2])

            point._offsets3d = (
                np.array([r[frame, 0]]),
                np.array([r[frame, 1]]),
                np.array([r[frame, 2]]),
            )

            if thrust_quiver is not None:
                thrust_quiver.remove()
                thrust_quiver = None

            if T_hist is not None:
                scale = 0.001
                thrust_quiver = ax.quiver(
                    [r[frame, 0]],
                    [r[frame, 1]],
                    [r[frame, 2]],
                    [scale * T_hist[frame, 0]],
                    [scale * T_hist[frame, 1]],
                    [scale * T_hist[frame, 2]],
                    color="C1",
                )

            # Add touchdown indicator in text
            touchdown_msg = ""
            if r[frame, 2] <= 0.1:
                touchdown_msg = "\n✓ TOUCHDOWN!"

            text.set_text(
                f"t = {t[frame]:5.2f} s\n"
                f"alt = {r[frame, 2]:7.2f} m\n"
                f"|v| = {np.linalg.norm(v[frame]):7.2f} m/s\n"
                f"m   = {m[frame]:7.2f} kg\n"
                f"T_z = {T_opt:7.2f} N{touchdown_msg}"
            )

            # Rotate view slowly
            azim = 45 + 0.3 * frame_idx
            ax.view_init(elev=25, azim=azim)

            return line, point, text

        interval = 1000 * (t[-1] / n_frames) / (1.0 / fps)
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=n_frames,
            init_func=init,
            blit=False,
            interval=interval,
        )

        # Save animation
        mp4_path = os.path.join(self.cfg.out_dir, f"{basename}.mp4")
        if animation.writers.is_available("ffmpeg"):
            ani.save(mp4_path, writer="ffmpeg", fps=fps, dpi=200)
            logger.info("Saved MP4 bisection touchdown animation '%s' using ffmpeg.", mp4_path)
        else:
            logger.warning(
                "ffmpeg unavailable; skipping animation '%s'. "
                "Install ffmpeg to enable MP4 export.",
                mp4_path,
            )
        plt.close(fig)

    def animate_comparison(
        self,
        classical: Dict[str, Array],
        convex: Dict[str, Array],
        basename: str = "comparison_animation",
        fps: int = 20,
    ) -> None:
        """Animate classical vs convex trajectories side‑by‑side.

        The left panel shows the classical bisection method trajectory; the right panel
        shows the convex SOCP trajectory. Both panels update in time with
        synchronized frames for direct visual comparison.
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        t_c = classical["t"]
        x_c = classical["x"]
        r_c = x_c[:, 0:3]

        t_o = convex["t"]
        r_o = convex["r"]

        # Use the maximum number of frames so both trajectories are shown
        # completely. The method with fewer samples simply holds its final
        # state once it has finished.
        n_frames = max(len(t_c), len(t_o))

        fig = plt.figure(figsize=(12, 5))
        ax_c = fig.add_subplot(1, 2, 1, projection="3d")
        ax_o = fig.add_subplot(1, 2, 2, projection="3d")

        # Common axis limits across both panels
        all_r = np.vstack([r_c, r_o])
        max_range = np.array(
            [
                all_r[:, 0].max() - all_r[:, 0].min(),
                all_r[:, 1].max() - all_r[:, 1].min(),
                all_r[:, 2].max() - all_r[:, 2].min(),
            ]
        ).max()
        mid_x = 0.5 * (all_r[:, 0].max() + all_r[:, 0].min())
        mid_y = 0.5 * (all_r[:, 1].max() + all_r[:, 1].min())
        mid_z = 0.5 * (all_r[:, 2].max() + all_r[:, 2].min())

        for ax in (ax_c, ax_o):
            ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
            ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
            ax.set_zlim(0.0, mid_z + max_range / 2)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_zlabel("z [m]")

        ax_c.set_title("Bisection Method")
        ax_o.set_title("Convex Optimization (SOCP)")

        # Pad outline
        theta = np.linspace(0, 2 * np.pi, 100)
        pad_radius = 10.0
        pad_x = pad_radius * np.cos(theta)
        pad_y = pad_radius * np.sin(theta)
        pad_z = np.zeros_like(theta)
        ax_c.plot(pad_x, pad_y, pad_z, color="k", linestyle="--")
        ax_o.plot(pad_x, pad_y, pad_z, color="k", linestyle="--")

        line_c, = ax_c.plot([], [], [], "C0-", lw=2)
        point_c = ax_c.scatter([], [], [], color="red", s=40)

        line_o, = ax_o.plot([], [], [], "C1-", lw=2)
        point_o = ax_o.scatter([], [], [], color="red", s=40)

        def init():
            line_c.set_data([], [])
            line_c.set_3d_properties([])
            line_o.set_data([], [])
            line_o.set_3d_properties([])
            return line_c, point_c, line_o, point_o

        def update(frame: int):
            # Clamp frame indices so that the shorter trajectory holds its
            # final state once finished, while the longer one keeps updating.
            idx_c = min(frame, len(r_c) - 1)
            idx_o = min(frame, len(r_o) - 1)

            # Classical
            line_c.set_data(r_c[: idx_c + 1, 0], r_c[: idx_c + 1, 1])
            line_c.set_3d_properties(r_c[: idx_c + 1, 2])
            point_c._offsets3d = (
                np.array([r_c[idx_c, 0]]),
                np.array([r_c[idx_c, 1]]),
                np.array([r_c[idx_c, 2]]),
            )

            # Convex
            line_o.set_data(r_o[: idx_o + 1, 0], r_o[: idx_o + 1, 1])
            line_o.set_3d_properties(r_o[: idx_o + 1, 2])
            point_o._offsets3d = (
                np.array([r_o[idx_o, 0]]),
                np.array([r_o[idx_o, 1]]),
                np.array([r_o[idx_o, 2]]),
            )

            # Slight synchronized rotation
            azim = 45 + 0.3 * frame
            for ax in (ax_c, ax_o):
                ax.view_init(elev=25, azim=azim)

            return line_c, point_c, line_o, point_o

        # Approximate frame interval based on the longer trajectory duration
        # so that both methods complete before the animation restarts.
        total_duration = max(t_c[-1], t_o[-1])
        interval = 1000 * (total_duration / n_frames) / (1.0 / fps)
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=n_frames,
            init_func=init,
            blit=False,
            interval=interval,
            repeat=True,
        )

        mp4_path = os.path.join(self.cfg.out_dir, f"{basename}.mp4")
        if animation.writers.is_available("ffmpeg"):
            ani.save(mp4_path, writer="ffmpeg", fps=fps, dpi=200)
            logger.info("Saved MP4 comparison animation '%s' using ffmpeg.", mp4_path)
        else:
            logger.warning(
                "ffmpeg unavailable; skipping comparison animation '%s'. "
                "Install ffmpeg to enable MP4 export.",
                mp4_path,
            )
        plt.close(fig)

    # ------------------------------------------------------------------
    # Dashboards and comparison plots
    # ------------------------------------------------------------------
    def comparison_dashboard(
        self,
        classical: Dict[str, Array],
        convex: Dict[str, Array],
        basename: str = "comparison_dashboard",
    ) -> None:
        """Side‑by‑side comparison of classical and convex solutions."""
        fig = plt.figure(figsize=(12, 8))

        # 3D trajectories
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        ax1 = fig.add_subplot(2, 3, 1, projection="3d")
        r_c = classical["x"][:, 0:3]
        r_o = convex["r"]
        ax1.plot(r_c[:, 0], r_c[:, 1], r_c[:, 2], label="Classical", color="C0")
        ax1.plot(r_o[:, 0], r_o[:, 1], r_o[:, 2], label="Convex", color="C1")
        ax1.set_title("3D Trajectory")
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        ax1.set_zlabel("z [m]")
        ax1.legend()

        # Velocity profiles
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(classical["t"], np.linalg.norm(classical["x"][:, 3:6], axis=1), label="Classical", color="C0")
        ax2.plot(convex["t"], np.linalg.norm(convex["v"], axis=1), label="Convex", color="C1")
        ax2.set_title("Speed vs Time")
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel("|v| [m/s]")
        ax2.grid(True)
        ax2.legend()

        # Thrust magnitude
        ax3 = fig.add_subplot(2, 3, 3)
        T_c = classical["T"]
        T_o = convex["T"]
        ax3.plot(classical["t"], np.linalg.norm(T_c, axis=1), label="Classical", color="C0")
        ax3.plot(convex["t"], np.linalg.norm(T_o, axis=1), label="Convex", color="C1")
        ax3.set_title("Thrust Magnitude")
        ax3.set_xlabel("t [s]")
        ax3.set_ylabel("|T| [N]")
        ax3.grid(True)
        ax3.legend()

        # Altitude‑velocity phase portrait
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(classical["x"][:, 2], classical["x"][:, 5], label="Classical", color="C0")
        ax4.plot(convex["r"][:, 2], convex["v"][:, 2], label="Convex", color="C1")
        ax4.set_title("v_z vs z")
        ax4.set_xlabel("z [m]")
        ax4.set_ylabel("v_z [m/s]")
        ax4.grid(True)
        ax4.legend()

        # Mass vs time
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(classical["t"], classical["x"][:, 6], label="Classical", color="C0")
        ax5.plot(convex["t"], convex["m"], label="Convex", color="C1")
        ax5.set_title("Mass vs Time")
        ax5.set_xlabel("t [s]")
        ax5.set_ylabel("m [kg]")
        ax5.grid(True)
        ax5.legend()

        # Fuel comparison bar plot
        ax6 = fig.add_subplot(2, 3, 6)
        fuel_classical = classical["x"][0, 6] - classical["x"][-1, 6]
        fuel_convex = convex["m"][0] - convex["m"][-1]
        ax6.bar(["Classical", "Convex"], [fuel_classical, fuel_convex], color=["C0", "C1"])
        ax6.set_title("Fuel Consumption Comparison")
        ax6.set_ylabel("Fuel [kg]")

        fig.tight_layout()
        self._save_figure(fig, basename)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Monte Carlo plots
    # ------------------------------------------------------------------
    def monte_carlo_plots(
        self,
        pos_errors: Array,
        vel_errors: Array,
        success_flags: Array,
        basename: str = "monte_carlo",
    ) -> None:
        """Scatter, histogram, and CDF plots for Monte Carlo results."""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # 2D scatter of landing position error magnitude vs trial index.
        ax0 = axes[0]
        idx = np.arange(pos_errors.size)
        ax0.scatter(idx[success_flags], pos_errors[success_flags], color="green", label="Success", s=10)
        ax0.scatter(idx[~success_flags], pos_errors[~success_flags], color="red", label="Failure", s=10)
        ax0.set_xlabel("Trial")
        ax0.set_ylabel("||r_land|| [m]")
        ax0.set_title("Landing Position Error by Trial")
        ax0.grid(True)
        ax0.legend()

        # Histogram of landing velocity errors.
        ax1 = axes[1]
        ax1.hist(vel_errors, bins=30, color="C0", alpha=0.8)
        ax1.set_xlabel("||v_land|| [m/s]")
        ax1.set_ylabel("Count")
        ax1.set_title("Landing Velocity Error Histogram")
        ax1.grid(True)

        # CDF plot.
        ax2 = axes[2]
        sorted_err = np.sort(pos_errors)
        cdf = np.linspace(0, 1, len(sorted_err), endpoint=True)
        ax2.plot(sorted_err, cdf, color="C1")
        ax2.axvline(5.0, color="k", linestyle="--", label="5 m threshold")
        ax2.set_xlabel("||r_land|| [m]")
        ax2.set_ylabel("P(error < x)")
        ax2.set_title("Landing Error CDF")
        ax2.grid(True)
        ax2.legend()

        fig.tight_layout()
        self._save_figure(fig, basename)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Convergence and energy plots
    # ------------------------------------------------------------------
    def bisection_convergence_plot(
        self,
        history: np.ndarray,
        basename: str = "bisection_convergence",
    ) -> None:
        """Plot bisection thrust estimate and |vz| vs iteration."""
        it = history[:, 0]
        Tz = history[:, 1]
        vz = history[:, 2]

        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax2 = ax1.twinx()

        ax1.plot(it, Tz, "C0-o", label="Tz estimate")
        ax2.plot(it, np.abs(vz), "C1-s", label="|v_z|")

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Tz [N]", color="C0")
        ax2.set_ylabel("|v_z| [m/s]", color="C1")
        ax1.set_title("Bisection Convergence")
        ax1.grid(True)

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")

        self._save_figure(fig, basename)
        plt.close(fig)

    def energy_plot(
        self,
        t: Array,
        ke: Array,
        pe: Array,
        etot: Array,
        basename: str = "energy_validation",
    ) -> None:
        """Plot kinetic, potential, and total energy vs time."""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, ke, label="KE", color="C0")
        ax.plot(t, pe, label="PE", color="C1")
        ax.plot(t, etot, label="Total", color="C2")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Energy [J]")
        ax.set_title("Energy Conservation Check")
        ax.grid(True)
        ax.legend()

        self._save_figure(fig, basename)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Bar charts and heatmaps
    # ------------------------------------------------------------------
    def comparison_bars(
        self,
        labels: Tuple[str, str],
        fuel: Tuple[float, float],
        accuracy: Tuple[float, float],
        time_s: Tuple[float, float],
        success_rate: Tuple[float, float],
        basename: str = "comparison_bars",
    ) -> None:
        """Comparison bar charts for fuel, accuracy, time, and success rate."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))

        x = np.arange(2)

        ax = axes[0, 0]
        ax.bar(x, fuel, color=["C0", "C1"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Fuel [kg]")
        ax.set_title("Fuel Consumption")

        ax = axes[0, 1]
        ax.bar(x, accuracy, color=["C0", "C1"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Landing Error [m]")
        ax.set_title("Landing Accuracy")

        ax = axes[1, 0]
        ax.bar(x, time_s, color=["C0", "C1"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Computation Time [s]")
        ax.set_title("Computation Time")

        ax = axes[1, 1]
        ax.bar(x, [100 * s for s in success_rate], color=["C0", "C1"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Success Rate [%]")
        ax.set_title("Monte Carlo Success Rate")

        fig.tight_layout()
        self._save_figure(fig, basename)
        plt.close(fig)

    def heatmap(
        self,
        x_vals: Array,
        y_vals: Array,
        Z: Array,
        xlabel: str,
        ylabel: str,
        title: str,
        basename: str,
    ) -> None:
        """Generic heatmap for experiments (e.g., fuel vs initial conditions)."""
        fig, ax = plt.subplots(figsize=(6, 5))
        X, Y = np.meshgrid(x_vals, y_vals, indexing="xy")
        cmap = plt.get_cmap(self.cfg.cmap)
        im = ax.pcolormesh(X, Y, Z, shading="auto", cmap=cmap)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        self._save_figure(fig, basename)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Convex Optimization Visualizations
    # ------------------------------------------------------------------
    def convex_optimization_dashboard(
        self,
        convex: Dict[str, Array],
        basename: str = "convex_dashboard",
    ) -> None:
        """Comprehensive dashboard for convex optimization results.
        
        Creates a multi-panel visualization showing:
        - 3D trajectory with thrust vectors
        - Thrust profile over time
        - Velocity components
        - Mass depletion and fuel efficiency
        - Phase portraits
        - Control effort visualization
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        
        t = convex["t"]
        r = convex["r"]
        v = convex["v"]
        m = convex["m"]
        T = convex["T"]
        u = convex.get("u", None)
        Gamma = convex.get("Gamma", None)
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. 3D Trajectory with thrust vectors (top left)
        ax1 = fig.add_subplot(3, 3, 1, projection="3d")
        ax1.plot(r[:, 0], r[:, 1], r[:, 2], "C1-", lw=2, label="Trajectory")
        ax1.scatter(r[0, 0], r[0, 1], r[0, 2], color="green", s=100, label="Start", marker="^")
        ax1.scatter(r[-1, 0], r[-1, 1], r[-1, 2], color="red", s=100, label="Landing", marker="s")
        
        # Thrust vectors (subsampled for clarity)
        idx = np.linspace(0, len(r) - 1, 15, dtype=int)
        scale = 0.001
        ax1.quiver(
            r[idx, 0], r[idx, 1], r[idx, 2],
            scale * T[idx, 0], scale * T[idx, 1], scale * T[idx, 2],
            color="orange", length=1.0, normalize=False, alpha=0.7
        )
        
        # Landing pad
        theta = np.linspace(0, 2 * np.pi, 100)
        pad_radius = 10.0
        ax1.plot(pad_radius * np.cos(theta), pad_radius * np.sin(theta), 
                np.zeros_like(theta), "k--", alpha=0.5, label="Landing pad")
        
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        ax1.set_zlabel("z [m]")
        ax1.set_title("3D Trajectory with Thrust Vectors")
        ax1.legend(loc="upper right", fontsize=8)
        
        # 2. Thrust magnitude profile (top middle)
        ax2 = fig.add_subplot(3, 3, 2)
        T_mag = np.linalg.norm(T, axis=1)
        ax2.plot(t, T_mag, "C1-", lw=2)
        ax2.axhline(6000, color="r", linestyle="--", alpha=0.5, label="T_min")
        ax2.axhline(15000, color="r", linestyle="--", alpha=0.5, label="T_max")
        ax2.fill_between(t, 6000, 15000, alpha=0.1, color="gray")
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel("|T| [N]")
        ax2.set_title("Thrust Magnitude Profile")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        
        # 3. Velocity components (top right)
        ax3 = fig.add_subplot(3, 3, 3)
        ax3.plot(t, v[:, 0], "C0-", label="v_x", lw=1.5)
        ax3.plot(t, v[:, 1], "C1-", label="v_y", lw=1.5)
        ax3.plot(t, v[:, 2], "C2-", label="v_z", lw=1.5)
        ax3.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax3.set_xlabel("t [s]")
        ax3.set_ylabel("Velocity [m/s]")
        ax3.set_title("Velocity Components")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        
        # 4. Mass depletion (middle left)
        ax4 = fig.add_subplot(3, 3, 4)
        fuel_used = m[0] - m
        fuel_efficiency = fuel_used / t  # kg/s
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(t, m, "C1-", lw=2, label="Mass")
        line2 = ax4_twin.plot(t, fuel_efficiency, "C2--", lw=1.5, label="Fuel rate")
        ax4.axhline(500, color="r", linestyle="--", alpha=0.5, label="Dry mass")
        ax4.set_xlabel("t [s]")
        ax4.set_ylabel("Mass [kg]", color="C1")
        ax4_twin.set_ylabel("Fuel Rate [kg/s]", color="C2")
        ax4.set_title("Mass Depletion & Fuel Efficiency")
        ax4.grid(True, alpha=0.3)
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc="upper right", fontsize=8)
        ax4.tick_params(axis='y', labelcolor='C1')
        ax4_twin.tick_params(axis='y', labelcolor='C2')
        
        # 5. Altitude-velocity phase portrait (middle middle)
        ax5 = fig.add_subplot(3, 3, 5)
        ax5.plot(r[:, 2], v[:, 2], "C1-", lw=2)
        ax5.scatter(r[0, 2], v[0, 2], color="green", s=100, marker="^", label="Start", zorder=5)
        ax5.scatter(r[-1, 2], v[-1, 2], color="red", s=100, marker="s", label="Landing", zorder=5)
        ax5.axvline(0, color="k", linestyle="--", alpha=0.3)
        ax5.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax5.set_xlabel("Altitude z [m]")
        ax5.set_ylabel("Vertical Velocity v_z [m/s]")
        ax5.set_title("Altitude-Velocity Phase Portrait")
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=8)
        
        # 6. Thrust direction (middle right) - polar plot of thrust angles
        ax6 = fig.add_subplot(3, 3, 6, projection="polar")
        T_xy = np.linalg.norm(T[:, 0:2], axis=1)
        T_z = T[:, 2]
        thrust_angles = np.arctan2(T_xy, T_z)  # Angle from vertical
        thrust_magnitudes = np.linalg.norm(T, axis=1)
        # Color by time
        scatter = ax6.scatter(thrust_angles, thrust_magnitudes, c=t, cmap="viridis", 
                             s=30, alpha=0.7, edgecolors="k", linewidths=0.5)
        ax6.set_theta_zero_location("N")
        ax6.set_theta_direction(-1)
        ax6.set_ylim(0, 16000)
        ax6.set_title("Thrust Direction (Polar)", pad=20)
        cbar = plt.colorbar(scatter, ax=ax6, pad=0.1)
        cbar.set_label("Time [s]")
        
        # 7. Speed profile (bottom left)
        ax7 = fig.add_subplot(3, 3, 7)
        speed = np.linalg.norm(v, axis=1)
        ax7.plot(t, speed, "C1-", lw=2)
        ax7.fill_between(t, 0, speed, alpha=0.3, color="C1")
        ax7.set_xlabel("t [s]")
        ax7.set_ylabel("Speed |v| [m/s]")
        ax7.set_title("Speed Profile")
        ax7.grid(True, alpha=0.3)
        
        # 8. Horizontal trajectory (bottom middle)
        ax8 = fig.add_subplot(3, 3, 8)
        ax8.plot(r[:, 0], r[:, 1], "C1-", lw=2)
        ax8.scatter(r[0, 0], r[0, 1], color="green", s=100, marker="^", label="Start", zorder=5)
        ax8.scatter(r[-1, 0], r[-1, 1], color="red", s=100, marker="s", label="Landing", zorder=5)
        # Landing pad circle
        circle = plt.Circle((0, 0), 10, fill=False, linestyle="--", color="k", alpha=0.5)
        ax8.add_patch(circle)
        ax8.set_xlabel("x [m]")
        ax8.set_ylabel("y [m]")
        ax8.set_title("Horizontal Trajectory (Top View)")
        ax8.set_aspect("equal")
        ax8.grid(True, alpha=0.3)
        ax8.legend(fontsize=8)
        
        # 9. Control effort (Gamma) if available (bottom right)
        ax9 = fig.add_subplot(3, 3, 9)
        if Gamma is not None:
            t_gamma = t[:-1]  # Gamma has N elements, t has N+1
            ax9.plot(t_gamma, Gamma, "C1-", lw=2, label="Γ (thrust/mass)")
            ax9.set_xlabel("t [s]")
            ax9.set_ylabel("Γ [N/kg]")
            ax9.set_title("Thrust-per-Mass (Control Variable)")
            ax9.grid(True, alpha=0.3)
            ax9.legend(fontsize=8)
        else:
            # Fallback: show acceleration magnitude
            accel = np.linalg.norm(v[1:] - v[:-1], axis=1) / (t[1] - t[0])
            ax9.plot(t[:-1], accel, "C1-", lw=2)
            ax9.set_xlabel("t [s]")
            ax9.set_ylabel("|a| [m/s²]")
            ax9.set_title("Acceleration Magnitude")
            ax9.grid(True, alpha=0.3)
        
        fig.suptitle("Convex Optimization (SOCP) - Comprehensive Analysis", 
                    fontsize=16, fontweight="bold", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        self._save_figure(fig, basename)
        plt.close(fig)

    def convex_thrust_heatmap(
        self,
        convex: Dict[str, Array],
        basename: str = "convex_thrust_heatmap",
    ) -> None:
        """Create a heatmap showing thrust magnitude over time with altitude overlay."""
        t = convex["t"]
        r = convex["r"]
        T = convex["T"]
        T_mag = np.linalg.norm(T, axis=1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Top: Thrust magnitude heatmap-style with altitude
        altitude = r[:, 2]
        
        # Create a 2D representation: time vs altitude with color = thrust
        scatter = ax1.scatter(t, altitude, c=T_mag, cmap="hot", s=50, edgecolors="k", linewidths=0.5)
        cbar1 = fig.colorbar(scatter, ax=ax1, label="Thrust [N]")
        ax1.plot(t, altitude, "k-", alpha=0.3, lw=1)
        ax1.set_ylabel("Altitude [m]")
        ax1.set_title("Thrust Magnitude vs Altitude (Color-coded)")
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        # Bottom: Thrust magnitude over time with v_z color-coding
        v = convex["v"]
        v_z = v[:, 2]
        scatter2 = ax2.scatter(t, T_mag, c=v_z, cmap="coolwarm", s=30, edgecolors="k", linewidths=0.5, zorder=3)
        ax2.plot(t, T_mag, "k-", lw=1, alpha=0.3, zorder=1)
        cbar2 = fig.colorbar(scatter2, ax=ax2, label="v_z [m/s]")
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel("|T| [N]")
        ax2.set_title("Thrust Profile Over Time (Color-coded by v_z)")
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        self._save_figure(fig, basename)
        plt.close(fig)
        
        # Create animated GIF for both plots
        v = convex["v"]
        v_z = v[:, 2]
        speed = np.linalg.norm(v, axis=1)
        altitude = r[:, 2]
        self._convex_thrust_animation(t, T_mag, v_z, altitude, speed, basename=f"{basename}_thrust_animation")
    
    def _convex_thrust_animation(
        self,
        t: Array,
        T_mag: Array,
        v_z: Array,
        altitude: Array,
        speed: Array,
        basename: str = "convex_thrust_animation",
        fps: int = 20,
    ) -> None:
        """Create animated GIF with 2 plots: Thrust vs Altitude and Thrust vs Time (color-coded by v_z)."""
        from matplotlib.collections import LineCollection
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)
        
        # Top plot: Thrust magnitude vs Altitude (color-coded by speed)
        ax1.set_xlim(altitude.min() * 1.05, altitude.max() * 0.95)
        ax1.set_ylim(T_mag.min() * 0.9, T_mag.max() * 1.1)
        ax1.set_xlabel("Altitude [m]")
        ax1.set_ylabel("|T| [N]")
        ax1.set_title("Thrust Magnitude vs Altitude (Color-coded by Speed)")
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()
        
        # Bottom plot: Thrust magnitude vs Time (color-coded by v_z)
        ax2.set_xlim(0, t[-1])
        ax2.set_ylim(T_mag.min() * 0.9, T_mag.max() * 1.1)
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel("|T| [N]")
        ax2.set_title("Thrust Profile Over Time (Color-coded by v_z)")
        ax2.grid(True, alpha=0.3)
        
        # Normalize for colormaps
        v_z_norm = (v_z - v_z.min()) / (v_z.max() - v_z.min() + 1e-10)
        speed_norm = (speed - speed.min()) / (speed.max() - speed.min() + 1e-10)
        cmap_vz = plt.cm.coolwarm
        cmap_speed = plt.cm.viridis
        
        # Create colorbars
        sm1 = plt.cm.ScalarMappable(cmap=cmap_speed, norm=plt.Normalize(vmin=speed.min(), vmax=speed.max()))
        sm1.set_array([])
        cbar1 = fig.colorbar(sm1, ax=ax1, label="Speed |v| [m/s]")
        
        sm2 = plt.cm.ScalarMappable(cmap=cmap_vz, norm=plt.Normalize(vmin=v_z.min(), vmax=v_z.max()))
        sm2.set_array([])
        cbar2 = fig.colorbar(sm2, ax=ax2, label="v_z [m/s]")
        
        # Subsample for smoother animation
        n_frames = min(150, len(t))
        frame_indices = np.linspace(0, len(t) - 1, n_frames, dtype=int)
        
        line_collection1 = None
        point1 = None
        line_collection2 = None
        point2 = None
        
        def init():
            nonlocal line_collection1, point1, line_collection2, point2
            line_collection1 = None
            point1 = None
            line_collection2 = None
            point2 = None
            return []
        
        def update(frame_idx: int):
            nonlocal line_collection1, point1, line_collection2, point2
            frame = frame_indices[frame_idx]
            
            # Remove previous collections
            if line_collection1 is not None:
                line_collection1.remove()
            if point1 is not None:
                point1.remove()
            if line_collection2 is not None:
                line_collection2.remove()
            if point2 is not None:
                point2.remove()
            
            if frame > 0:
                # Top plot: Thrust vs Altitude (color by speed)
                segments1 = []
                colors1 = []
                for i in range(frame):
                    segments1.append([(altitude[i], T_mag[i]), (altitude[i+1], T_mag[i+1])])
                    colors1.append(cmap_speed(speed_norm[i]))
                
                if segments1:
                    line_collection1 = LineCollection(segments1, colors=colors1, linewidths=2, alpha=0.9)
                    ax1.add_collection(line_collection1)
                
                # Add current point
                color1 = cmap_speed(speed_norm[frame])
                point1 = ax1.scatter(altitude[frame], T_mag[frame], c=[color1], s=150, 
                                    edgecolors="k", linewidths=2, zorder=5)
                
                # Bottom plot: Thrust vs Time (color by v_z)
                segments2 = []
                colors2 = []
                for i in range(frame):
                    segments2.append([(t[i], T_mag[i]), (t[i+1], T_mag[i+1])])
                    colors2.append(cmap_vz(v_z_norm[i]))
                
                if segments2:
                    line_collection2 = LineCollection(segments2, colors=colors2, linewidths=2, alpha=0.9)
                    ax2.add_collection(line_collection2)
                
                # Add current point
                color2 = cmap_vz(v_z_norm[frame])
                point2 = ax2.scatter(t[frame], T_mag[frame], c=[color2], s=150, 
                                    edgecolors="k", linewidths=2, zorder=5)
            
            return []
        
        interval = 1000 * (t[-1] / n_frames) / (1.0 / fps)
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=n_frames,
            init_func=init,
            blit=False,
            interval=interval,
            repeat=True,
        )
        
        # Save as GIF
        gif_path = os.path.join(self.cfg.out_dir, f"{basename}.gif")
        try:
            ani.save(gif_path, writer="pillow", fps=fps, dpi=100)
            logger.info("Saved GIF animation '%s'", gif_path)
        except Exception as e:
            logger.warning("Failed to save GIF animation '%s': %s", gif_path, e)
        
        plt.close(fig)

    def convex_phase_space(
        self,
        convex: Dict[str, Array],
        basename: str = "convex_phase_space",
    ) -> None:
        """Multi-panel phase space analysis of convex optimization trajectory."""
        r = convex["r"]
        v = convex["v"]
        t = convex["t"]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Altitude vs vertical velocity
        ax = axes[0, 0]
        ax.plot(r[:, 2], v[:, 2], "C1-", lw=2)
        ax.scatter(r[0, 2], v[0, 2], color="green", s=150, marker="^", 
                  label="Start", zorder=5, edgecolors="k", linewidths=1)
        ax.scatter(r[-1, 2], v[-1, 2], color="red", s=150, marker="s", 
                  label="Landing", zorder=5, edgecolors="k", linewidths=1)
        # Color-code by time
        points = ax.scatter(r[:, 2], v[:, 2], c=t, cmap="viridis", s=20, alpha=0.6, zorder=3)
        ax.axvline(0, color="k", linestyle="--", alpha=0.3)
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Altitude z [m]")
        ax.set_ylabel("Vertical Velocity v_z [m/s]")
        ax.set_title("Altitude-Velocity Phase Space")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.colorbar(points, ax=ax, label="Time [s]")
        
        # 2. Horizontal position vs horizontal velocity
        ax = axes[0, 1]
        r_horiz = np.linalg.norm(r[:, 0:2], axis=1)
        v_horiz = np.linalg.norm(v[:, 0:2], axis=1)
        ax.plot(r_horiz, v_horiz, "C1-", lw=2)
        ax.scatter(r_horiz[0], v_horiz[0], color="green", s=150, marker="^", 
                  label="Start", zorder=5, edgecolors="k", linewidths=1)
        ax.scatter(r_horiz[-1], v_horiz[-1], color="red", s=150, marker="s", 
                  label="Landing", zorder=5, edgecolors="k", linewidths=1)
        points = ax.scatter(r_horiz, v_horiz, c=t, cmap="viridis", s=20, alpha=0.6, zorder=3)
        ax.set_xlabel("Horizontal Distance ||r_xy|| [m]")
        ax.set_ylabel("Horizontal Speed ||v_xy|| [m/s]")
        ax.set_title("Horizontal Phase Space")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.colorbar(points, ax=ax, label="Time [s]")
        
        # 3. Speed vs altitude
        ax = axes[1, 0]
        speed = np.linalg.norm(v, axis=1)
        ax.plot(r[:, 2], speed, "C1-", lw=2)
        ax.scatter(r[0, 2], speed[0], color="green", s=150, marker="^", 
                  label="Start", zorder=5, edgecolors="k", linewidths=1)
        ax.scatter(r[-1, 2], speed[-1], color="red", s=150, marker="s", 
                  label="Landing", zorder=5, edgecolors="k", linewidths=1)
        points = ax.scatter(r[:, 2], speed, c=t, cmap="viridis", s=20, alpha=0.6, zorder=3)
        ax.set_xlabel("Altitude z [m]")
        ax.set_ylabel("Speed |v| [m/s]")
        ax.set_title("Speed vs Altitude")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.invert_xaxis()
        plt.colorbar(points, ax=ax, label="Time [s]")
        
        # 4. Energy vs altitude
        ax = axes[1, 1]
        m = convex["m"]
        ke = 0.5 * m * speed**2
        pe = m * 9.81 * r[:, 2]  # g = 9.81
        etot = ke + pe
        ax.plot(r[:, 2], ke, "C0-", lw=1.5, label="KE", alpha=0.7)
        ax.plot(r[:, 2], pe, "C1-", lw=1.5, label="PE", alpha=0.7)
        ax.plot(r[:, 2], etot, "C2-", lw=2, label="Total Energy")
        ax.set_xlabel("Altitude z [m]")
        ax.set_ylabel("Energy [J]")
        ax.set_title("Energy vs Altitude")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.invert_xaxis()
        
        fig.suptitle("Convex Optimization - Phase Space Analysis", 
                    fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        self._save_figure(fig, basename)
        plt.close(fig)


__all__ = ["VisualConfig", "RocketVisualizer"]


