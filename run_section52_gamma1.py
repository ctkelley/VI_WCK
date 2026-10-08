"""Figure 3 and Figure 4: the semilinear elliptic VI with gamma=1."""
from pathlib import Path

import numpy as np

from experiments.section52_runner import run_section52_case
from src.section52_problem import (
    F52,
    build_problem,
    make_boundary_manufactured_solution,
)

OUTPUT = Path(__file__).resolve().parent / "results" / "section52_gamma1"


def run_case(name, problem):
    outdir = OUTPUT / name
    R0 = run_section52_case(problem, outdir)
    with (outdir / "parameters.txt").open("w") as handle:
        handle.write(f"gamma = {problem.gamma:.17e}\n")
        handle.write(f"gamma_star = {problem.gamma_star:.17e}\n")
        handle.write(f"gamma_bar = {problem.gamma_bar:.17e}\n")
        handle.write(f"mu = {problem.mu:.17e}\n")
        handle.write(f"R0 = {R0:.17e}\n")
        count = np.count_nonzero(np.abs(problem.xstar) == problem.l)
        handle.write(f"boundary_components = {count}\n")
        handle.write(f"norm_F_xstar_2 = {np.linalg.norm(F52(problem.xstar, problem)):.17e}\n")
    print(f"{name}: gamma_star={problem.gamma_star:.5g}, "
          f"mu={problem.mu:.5g}, R0={R0:.5g}")


def main():
    settings = dict(N=15, l=1.0, p=6.0, q=0.75, beta=0.15, gamma=1.0)
    interior = build_problem(**settings, amplitude=0.35)
    run_case("interior", interior)

    xstar, wstar, _ = make_boundary_manufactured_solution(
        N=15, l=1.0, amplitude=0.35, normal_magnitude=0.5,
    )
    boundary = build_problem(**settings, xstar=xstar, wstar=wstar)
    run_case("boundary", boundary)


if __name__ == "__main__":
    main()
