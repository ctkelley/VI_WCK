import csv
from pathlib import Path
import numpy as np


def save_history(path, output):
    path = Path(path)
    initial_residual = output.residual_history[0]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "distance_to_xstar", "fixed_point_residual", "relative_residual"])
        for k, (error, residual) in enumerate(zip(output.error_history, output.residual_history)):
            relative = residual / initial_residual if initial_residual != 0.0 else 0.0
            writer.writerow([k, f"{error:.17e}", f"{residual:.17e}", f"{relative:.17e}"])


def save_summary(path, rows):
    path = Path(path)
    fields = [
        "tau",
        "method",
        "m",
        "iterations",
        "max_iterations",
        "converged",
        "target_relative_residual",
        "final_error",
        "minimum_error",
        "final_relative_residual",
        "accepted",
        "rejected",
        "acceptance_rate",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_row(tau, method, m, output):
    return {
        "tau": tau,
        "method": method,
        "m": m,
        "iterations": output.iterations,
        "max_iterations": output.max_iterations,
        "converged": output.converged,
        "target_relative_residual": f"{output.target_relative_residual:.17e}",
        "final_error": f"{output.error_history[-1]:.17e}",
        "minimum_error": f"{np.min(output.error_history):.17e}",
        "final_relative_residual": f"{output.final_relative_residual:.17e}",
        "accepted": "" if len(output.accepted_history) == 0 else output.accepted,
        "rejected": "" if len(output.accepted_history) == 0 else output.rejected,
        "acceptance_rate": "" if len(output.accepted_history) == 0 else f"{output.acceptance_rate:.17e}",
    }
