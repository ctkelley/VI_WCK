"""Figure 2 and Table 1: Example 3 safeguard comparison."""
from pathlib import Path
import csv
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.io_utils import save_history
from src.algorithms import algorithm1, aa_without_safeguard, algorithm2
from src.example3_problem import XSTAR, example3_R0, make_example3_operator, projection_nonnegative
from src.plotting import plot_distance_curves

SIGMA = 0.7
TAU = 7.0e-4
M = 1
NU = 0.01
RHO = 0.999
X0 = np.array([1.3, 0.2], dtype=float)
MAX_ITERATIONS = 10000
TOL = 1.0e-8


def main():
    outdir = ROOT / "results" / "example3"
    outdir.mkdir(parents=True, exist_ok=True)
    F = make_example3_operator(SIGMA)
    R0 = example3_R0(F, X0, SIGMA)

    basic = algorithm1(
        F, projection_nonnegative, X0, TAU, XSTAR,
        max_iterations=MAX_ITERATIONS, target_relative_residual=TOL,
    )
    aa = aa_without_safeguard(
        F, projection_nonnegative, X0, TAU, M, XSTAR,
        max_iterations=MAX_ITERATIONS, target_relative_residual=TOL,
    )
    paa = algorithm2(
        F, projection_nonnegative, X0, TAU, M, NU, RHO, R0, XSTAR,
        max_iterations=MAX_ITERATIONS, target_relative_residual=TOL,
    )

    runs = [
        ("Algorithm 1", "algorithm1", basic),
        ("AA without safeguard (m=1)", "aa_without_safeguard_m1", aa),
        ("Algorithm 2 (m=1)", "algorithm2_m1", paa),
    ]
    for _, suffix, result in runs:
        save_history(outdir / f"history_{suffix}.csv", result)
    plot_distance_curves([(label, result) for label, _, result in runs], outdir / "distance_comparison.png")

    with (outdir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "method", "m", "iterations", "max_iterations", "converged",
            "target_relative_residual", "final_error", "minimum_error",
            "final_relative_residual", "accepted", "rejected", "acceptance_rate",
        ])
        for index, (label, _, result) in enumerate(runs):
            safeguarded = index == 2
            writer.writerow([
                label, 0 if index == 0 else M, result.iterations,
                result.max_iterations, result.converged, result.target_relative_residual,
                result.error_history[-1], np.min(result.error_history),
                result.final_relative_residual,
                result.accepted if safeguarded else "",
                result.rejected if safeguarded else "",
                result.acceptance_rate if safeguarded else "",
            ])

    with (outdir / "parameters.txt").open("w") as handle:
        handle.write(f"sigma = {SIGMA}\ntau = {TAU}\nx0 = {X0.tolist()}\n")
        handle.write(f"m = {M}\nnu = {NU}\nrho = {RHO}\n")
        handle.write(f"R0 = {R0:.17e}\nRbar = {paa.Rbar:.17e}\n")
        handle.write(f"relative_residual_tolerance = {TOL}\nmax_iterations = {MAX_ITERATIONS}\n")

    for label, _, result in runs:
        print(f"{label}: k={result.iterations}, converged={result.converged}, "
              f"relative_residual={result.final_relative_residual:.4e}")
    print(f"Algorithm 2: accepted={paa.accepted}, rejected={paa.rejected}, "
          f"rate={paa.acceptance_rate:.2%}")


if __name__ == "__main__":
    main()
