"""Run the two Section 6.2 step sizes for one manufactured solution."""
from pathlib import Path

import numpy as np

from src.algorithms import algorithm1, algorithm2
from src.plotting import plot_distance_curves
from src.section52_problem import F52, R0_section52, project_box
from experiments.io_utils import save_history, save_summary, summary_row

# Figure 3 and Figure 4 in the paper.
TAUS = (1.0e-4, 1.0e-3)
M_VALUES = (1, 2, 3, 5)
NU = 100.0
RHO = 0.99
MAX_ITERATIONS = 10000
TARGET_RELATIVE_RESIDUAL = 1.0e-8


def run_section52_case(data, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    x0 = np.zeros_like(data.xstar)
    R0 = R0_section52(data, x0)
    project = lambda x: project_box(x, data.l)
    F = lambda x: F52(x, data)
    summary = []

    for tau in TAUS:
        tag = f"{tau:.0e}".replace("-", "m")
        basic = algorithm1(
            F, project, x0, tau, data.xstar,
            max_iterations=MAX_ITERATIONS,
            target_relative_residual=TARGET_RELATIVE_RESIDUAL,
        )
        save_history(outdir / f"history_tau_{tag}_algorithm1.csv", basic)
        summary.append(summary_row(tau, "Algorithm 1", 0, basic))
        curves = [("Algorithm 1", basic)]

        for m in M_VALUES:
            result = algorithm2(
                F, project, x0, tau, m, NU, RHO, R0, data.xstar,
                max_iterations=MAX_ITERATIONS,
                target_relative_residual=TARGET_RELATIVE_RESIDUAL,
            )
            save_history(outdir / f"history_tau_{tag}_algorithm2_m{m}.csv", result)
            summary.append(summary_row(tau, f"Algorithm 2 (m={m})", m, result))
            curves.append((f"Algorithm 2 (m={m})", result))

        plot_distance_curves(curves, outdir / f"distance_tau_{tag}.png")

    save_summary(outdir / "summary.csv", summary)
    return R0
