"""Plot Euclidean errors against iteration counts (Figures 2--4)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLOT_FLOOR = 1.0e-16


def plot_distance_curves(outputs, path):
    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    for label, output in outputs:
        distance = np.maximum(output.error_history, PLOT_FLOOR)
        ax.semilogy(np.arange(distance.size), distance, label=label, linewidth=2.2)

    ax.set_xlabel("Iteration k", fontsize=19)
    ax.set_ylabel(r"Distance $\|x_k-x^*\|_2$", fontsize=19)
    ax.tick_params(axis="both", which="major", labelsize=17)
    ax.tick_params(axis="both", which="minor", labelsize=15)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=220, bbox_inches="tight")
    plt.close(fig)
