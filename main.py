"""Entry point for numerical optimization of rocket landing trajectories.

This script orchestrates classical and convex optimization solvers,
generates trajectories, runs experiments, and produces visualizations.

Usage examples:

    python main.py --method both --experiments all --save-animations
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict

import numpy as np

from classical_solver import ClassicalSolver
from convex_solver import ConvexSolver
from dynamics import Rocket3D, RocketConfig, default_initial_state
from experiments import ExperimentRunner
from visualization import RocketVisualizer


def setup_logging() -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def run_solvers(
    method: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run classical and/or convex solvers and return trajectories."""
    rocket = Rocket3D()
    classical = ClassicalSolver(rocket)
    convex = ConvexSolver()

    cfg = rocket.config
    x0 = default_initial_state(cfg)

    results: Dict[str, Dict[str, np.ndarray]] = {}

    if method in ("classical", "both"):
        # Constant thrust via bisection - finds thrust that actually lands
        bis = classical.constant_thrust_bisection()
        T_vec = np.array([0.0, 0.0, bis.T_opt])
        
        # Get full landing trajectory
        t_c, x_c = classical.find_landing_trajectory(bis.T_opt)

        # Compute thrust history
        T_hist = np.zeros((t_c.size, 3))
        T_hist[:] = T_vec[None, :]

        # Convert history format for compatibility
        bisection_history = np.array([(h[0], h[1], h[2]) for h in bis.history])

        results["classical"] = {
            "t": t_c,
            "x": x_c,
            "T": T_hist,
            "bisection_history": bisection_history,
        }

    if method in ("convex", "both"):
        # Convex SOCP
        rcfg = RocketConfig()
        x0c = default_initial_state(rcfg)
        r0 = x0c[0:3]
        v0 = x0c[3:6]
        m0 = x0c[6]

        convex.formulate_socp(r0=r0, v0=v0, m0=m0)
        success, msg = convex.solve(verbose=False)
        if not success:
            raise RuntimeError(f"Convex solver failed: {msg}")
        traj = convex.extract_trajectory()
        results["convex"] = traj

    return results


def run_experiments(do_experiments: bool) -> None:
    """Run experiment suite if requested."""
    if not do_experiments:
        return

    rocket = Rocket3D()
    classical = ClassicalSolver(rocket)
    runner = ExperimentRunner(rocket=rocket, classical_solver=classical)

    runner.experiment_wind_disturbance()
    runner.experiment_landing_offset()
    runner.experiment_initial_velocity_heatmap()
    runner.experiment_engine_failure()
    runner.experiment_discretization_study()


def main() -> None:
    """Parse arguments and orchestrate simulations and plots."""
    parser = argparse.ArgumentParser(
        description="Numerical Optimization of Rocket Landing Trajectories"
    )
    parser.add_argument(
        "--method",
        choices=["classical", "convex", "both"],
        default="both",
        help="Which solver(s) to run.",
    )
    parser.add_argument(
        "--experiments",
        choices=["none", "all"],
        default="all",
        help="Whether to run experiment suite.",
    )
    parser.add_argument(
        "--save-animations",
        action="store_true",
        help="If set, save 3D descent animation MP4.",
    )
    parser.add_argument(
        "--bisection-animation",
        action="store_true",
        help="If set, generate bisection touchdown animation (runs until touchdown).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to store figures and results.",
    )

    args = parser.parse_args()

    setup_logging()

    os.makedirs(args.output_dir, exist_ok=True)

    # Run solvers
    results = run_solvers(args.method)

    # Visualizations
    from visualization import VisualConfig

    vis = RocketVisualizer(VisualConfig(out_dir=args.output_dir))

    if "classical" in results:
        c = results["classical"]
        # Energy plot for classical trajectory
        ke, pe, etot = Rocket3D().energy_history(c["x"])
        vis.energy_plot(c["t"], ke, pe, etot, basename="energy_classical")

    if "convex" in results:
        o = results["convex"]
        vis.plot_3d_trajectory(o["t"], o["r"], T=o["T"], basename="trajectory_convex")
        # Generate comprehensive convex optimization visualizations
        vis.convex_optimization_dashboard(o, basename="convex_dashboard")
        vis.convex_thrust_heatmap(o, basename="convex_thrust_heatmap")
        vis.convex_phase_space(o, basename="convex_phase_space")

    if "classical" in results and "convex" in results:
        vis.comparison_dashboard(results["classical"], results["convex"])

        # Bisection convergence
        hist = results["classical"]["bisection_history"]
        vis.bisection_convergence_plot(hist)

    # Optional animations
    if args.save_animations:
        # Classical-only animation (Bisection method)
        if "classical" in results:
            c = results["classical"]
            r_c = c["x"][:, 0:3]
            v_c = c["x"][:, 3:6]
            m_c = c["x"][:, 6]
            vis.animate_descent(
                c["t"],
                r_c,
                v_c,
                m_c,
                T=c["T"],
                basename="descent_classical",
                title="Bisection Method",
            )

        # Convex-only animation and side-by-side comparison
        if "convex" in results:
            o = results["convex"]
            vis.animate_descent(o["t"], o["r"], o["v"], o["m"], T=o["T"], title="Convex Optimization (SOCP)")
            if "classical" in results:
                vis.animate_comparison(results["classical"], results["convex"])

    # Bisection touchdown animation (runs until touchdown)
    if args.bisection_animation:
        if "classical" in results:
            rocket = Rocket3D()
            c = results["classical"]
            # Get the optimal thrust from bisection history
            # The last entry in bisection_history has the final T_opt
            bis_hist = c["bisection_history"]
            T_opt = bis_hist[-1, 1]  # Second column is Tz
            vis.animate_bisection_touchdown(
                rocket,
                T_opt,
                basename="bisection_touchdown_animation",
            )
        else:
            # Run bisection if not already done
            rocket = Rocket3D()
            classical = ClassicalSolver(rocket)
            bis = classical.constant_thrust_bisection(t_final=40.0)
            vis.animate_bisection_touchdown(
                rocket,
                bis.T_opt,
                basename="bisection_touchdown_animation",
            )

    # Run experiments if requested
    run_experiments(args.experiments == "all")


if __name__ == "__main__":
    main()


