import math
import numpy as np


P = np.array([[2.0, 1.0], [1.0, 1.0]])
Q = np.array([[0.0, -1.0], [1.0, 0.0]])
b = np.array([1.0, 1.0])
XSTAR = np.array([0.0, 1.0])


def projection_nonnegative(x):
    """Projection onto Omega = R_+^2."""
    return np.maximum(np.asarray(x, dtype=float), 0.0)


def make_example3_operator(sigma):
    """Return the exact operator F from Example 3 in the paper."""
    if not (0.0 < sigma < 1.0):
        raise ValueError("sigma must be in (0, 1).")

    def F(x):
        x = np.asarray(x, dtype=float)
        y = P @ x - b
        r = float(np.linalg.norm(y))

        if r == 0.0:
            f1_core = np.zeros(2, dtype=float)
        else:
            f1_core = r ** (sigma - 1.0) * (P.T @ y)

        f1 = f1_core + np.array([1.0 / 8.0, 0.0])
        f2 = 16.0 * r**2 / (1.0 + r**2) * (P.T @ (Q @ y))
        return f1 + f2

    return F


def example3_mu(sigma):
    """The paper's sigma-star strong monotonicity constant for Example 3."""
    lambda_min_P = (3.0 - math.sqrt(5.0)) / 2.0
    return lambda_min_P ** (1.0 + sigma)


def example3_R0(F, x0, sigma):
    mu = example3_mu(sigma)
    return (np.linalg.norm(F(x0)) / mu) ** (1.0 / sigma)
