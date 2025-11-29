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
    ) -> None:
        """Generate an MP4 animation of the descent."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

        line, = ax.plot([], [], [], "C0-", lw=2)
        point = ax.scatter([], [], [], color="red", s=50)
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
        ax.set_title("3D Powered Descent Animation")

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

    def animate_comparison(
        self,
        classical: Dict[str, Array],
        convex: Dict[str, Array],
        basename: str = "comparison_animation",
        fps: int = 20,
    ) -> None:
        """Animate classical vs convex trajectories side‑by‑side.

        The left panel shows the classical RK4 trajectory; the right panel
        shows the convex SOCP trajectory. Both panels update in time with
        synchronized frames for direct visual comparison.
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        t_c = classical["t"]
        x_c = classical["x"]
        r_c = x_c[:, 0:3]

        t_o = convex["t"]
        r_o = convex["r"]

        # Use the minimum number of frames so both remain within bounds.
        n_frames = min(len(t_c), len(t_o))

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

        ax_c.set_title("Classical (RK4 + Bisection)")
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
            # Classical
            line_c.set_data(r_c[: frame + 1, 0], r_c[: frame + 1, 1])
            line_c.set_3d_properties(r_c[: frame + 1, 2])
            point_c._offsets3d = (
                np.array([r_c[frame, 0]]),
                np.array([r_c[frame, 1]]),
                np.array([r_c[frame, 2]]),
            )

            # Convex
            line_o.set_data(r_o[: frame + 1, 0], r_o[: frame + 1, 1])
            line_o.set_3d_properties(r_o[: frame + 1, 2])
            point_o._offsets3d = (
                np.array([r_o[frame, 0]]),
                np.array([r_o[frame, 1]]),
                np.array([r_o[frame, 2]]),
            )

            # Slight synchronized rotation
            azim = 45 + 0.3 * frame
            for ax in (ax_c, ax_o):
                ax.view_init(elev=25, azim=azim)

            return line_c, point_c, line_o, point_o

        # Approximate frame interval based on classical trajectory duration.
        interval = 1000 * (t_c[-1] / n_frames) / (1.0 / fps)
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=n_frames,
            init_func=init,
            blit=False,
            interval=interval,
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


__all__ = ["VisualConfig", "RocketVisualizer"]


