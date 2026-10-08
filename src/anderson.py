import numpy as np


def minimum_norm_least_squares(A, b):
    """Return the minimum-norm SVD solution of min ||A y - b||_2."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.shape[1] == 0:
        return np.array([], dtype=float)

    U, singular_values, Vt = np.linalg.svd(A, full_matrices=False)
    sigma_max = float(np.max(singular_values)) if singular_values.size else 0.0
    tolerance = max(A.shape) * np.finfo(float).eps * max(1.0, sigma_max)

    coefficients = U.T @ b
    coefficients = np.array(
        [value / sigma if sigma > tolerance else 0.0
         for value, sigma in zip(coefficients, singular_values)],
        dtype=float,
    )
    return Vt.T @ coefficients


def anderson_coefficients(g_history):
    """Solve min ||sum_j lambda_j g_j||_2 subject to sum_j lambda_j = 1."""
    if len(g_history) < 1:
        raise ValueError("At least one residual vector is required.")
    if len(g_history) == 1:
        return np.array([1.0], dtype=float)

    G = np.column_stack(g_history)
    g_reference = G[:, -1]
    D = G[:, :-1] - g_reference[:, None]
    y = minimum_norm_least_squares(D, -g_reference)
    return np.concatenate([y, np.array([1.0 - np.sum(y)])])
