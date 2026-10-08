# Projection Anderson Acceleration Algorithm

Python code for the experiments in Section 6 of *A Projection Anderson Acceleration Algorithm for Non-quasi-monotone Variational Inequalities with H{\"o}lder Continuous Terms* by X. Chen, C. T. Kelley, and L. Wang.

## Experiments

### Section 6.1: Example 3 (Figure 2 and Table 1)

The two-dimensional complementarity problem is solved by the projection method (Algorithm 1), Anderson acceleration without safeguard, and safeguarded PAA (Algorithm 2).

| Parameter | Value |
| --- | --- |
| `sigma` | `0.7` |
| `x0` | `(1.3, 0.2)` |
| `tau` | `7e-4` |
| Anderson depth | `m = 1` |
| Safeguard | `nu = 0.01`, `rho = 0.999` |

### Section 6.2: semilinear elliptic VI (Figures 3 and 4)

The problem is discretized on a `15 x 15` interior grid, with `n = 225`, box radius `l = 1`, `p = 6`, `q = 0.75`, skew coefficient `beta = 0.15`, and `gamma = 1`.

- Initial point: `x0 = 0`.
- Step sizes: `tau = 1e-4` and `tau = 1e-3`.
- PAA depths: `m = 1, 2, 3, 5` (plus Algorithm 1).
- Safeguard: `nu = 100`, `rho = 0.99`.
- Interior solution: equation (6.3), with amplitude `0.35`.
- Boundary solution: first/last grid columns fixed at `+1/-1`, with normal components `+0.5/-0.5` as in equation (6.4).

All runs stop at a relative fixed-point residual of `1e-8` or after `10000` iterations:

```text
||proj(x_k - tau F(x_k)) - x_k||_inf
----------------------------------- <= 1e-8.
||proj(x_0 - tau F(x_0)) - x_0||_inf
```

Figures plot the Euclidean distance `||x_k - x^*||_2`. 

The associated CSV histories and summary files are included for reproducing the results.

## Run

Python 3.10 or newer is required. From the project root, install the dependencies and run:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

To run a single experiment:

```bash
python experiments/run_example3.py
python run_section52_gamma1.py
```

## Files

- `src/algorithms.py`: Algorithm 1, AA without safeguard, and Algorithm 2.
- `src/anderson.py`: constrained Anderson least squares, using SVD.
- `src/example3_problem.py`: operator and projection for Example 3.
- `src/section52_problem.py`: elliptic VI, finite-difference matrices, and manufactured solutions.
- `src/plotting.py`: distance-to-solution convergence plots.
- `experiments/`: experiment drivers and CSV export.
- `run_all.py`: reproduce Figure 2, Figure 3, and Figure 4.
- `results/example3/`: Figure 2 image, Table 1 summary, and histories.
- `results/section52_gamma1/{interior,boundary}/`: Figures 3 and 4 (both step sizes), summaries, and histories.

