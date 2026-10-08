from dataclasses import dataclass
import numpy as np

from .anderson import anderson_coefficients


@dataclass
class IterationResult:
    solution: np.ndarray
    residual_history: np.ndarray
    error_history: np.ndarray
    accepted_history: np.ndarray
    max_iterations: int
    iterations: int
    converged: bool
    target_relative_residual: float
    R0: float | None = None
    Rbar: float | None = None

    @property
    def accepted(self):
        return int(np.sum(self.accepted_history))

    @property
    def rejected(self):
        return int(len(self.accepted_history) - np.sum(self.accepted_history))

    @property
    def acceptance_rate(self):
        if len(self.accepted_history) == 0:
            return float("nan")
        return float(np.mean(self.accepted_history))

    @property
    def initial_residual(self):
        return float(self.residual_history[0])

    @property
    def final_residual(self):
        return float(self.residual_history[-1])

    @property
    def final_relative_residual(self):
        if self.initial_residual == 0.0:
            return 0.0
        return self.final_residual / self.initial_residual


def _record_residual(F, projection, x, tau):
    z = projection(x - tau * F(x))
    g = z - x
    return z, g, float(np.linalg.norm(g, ord=np.inf))


def _stopping_threshold(initial_residual, target_relative_residual):
    return target_relative_residual * initial_residual


def algorithm1(
    F,
    projection,
    x0,
    tau,
    xstar,
    max_iterations=10000,
    target_relative_residual=1.0e-8,
):
    """Algorithm 1 with relative fixed-point-residual stopping."""
    if tau <= 0.0:
        raise ValueError("tau must be positive.")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative.")
    if target_relative_residual < 0.0:
        raise ValueError("target_relative_residual must be nonnegative.")

    x = projection(np.asarray(x0, dtype=float)).copy()
    residuals = []
    errors = []
    converged = False

    for k in range(max_iterations + 1):
        z, _, residual = _record_residual(F, projection, x, tau)
        residuals.append(residual)
        errors.append(float(np.linalg.norm(x - xstar)))

        if k == 0:
            threshold = _stopping_threshold(residual, target_relative_residual)

        if residual <= threshold:
            converged = True
            break
        if k == max_iterations:
            break
        x = z

    iterations = len(residuals) - 1
    return IterationResult(
        solution=x,
        residual_history=np.asarray(residuals, dtype=float),
        error_history=np.asarray(errors, dtype=float),
        accepted_history=np.array([], dtype=bool),
        max_iterations=max_iterations,
        iterations=iterations,
        converged=converged,
        target_relative_residual=target_relative_residual,
    )


def aa_without_safeguard(
    F,
    projection,
    x0,
    tau,
    m,
    xstar,
    max_iterations=10000,
    target_relative_residual=1.0e-8,
):
    """Projected Anderson acceleration without the Algorithm 2 safeguard."""
    if m < 0:
        raise ValueError("m must be nonnegative.")
    if tau <= 0.0:
        raise ValueError("tau must be positive.")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative.")
    if target_relative_residual < 0.0:
        raise ValueError("target_relative_residual must be nonnegative.")

    x = projection(np.asarray(x0, dtype=float)).copy()
    z_history = []
    g_history = []
    residuals = []
    errors = []
    converged = False

    for k in range(max_iterations + 1):
        z, g, residual = _record_residual(F, projection, x, tau)
        residuals.append(residual)
        errors.append(float(np.linalg.norm(x - xstar)))

        if k == 0:
            threshold = _stopping_threshold(residual, target_relative_residual)

        if residual <= threshold:
            converged = True
            break
        if k == max_iterations:
            break

        z_history.append(z.copy())
        g_history.append(g.copy())
        if len(z_history) > m + 1:
            z_history.pop(0)
            g_history.pop(0)

        # Coefficients sum to one; the resulting point is projected onto Omega.
        lam = anderson_coefficients(g_history)
        trial = np.zeros_like(x)
        for coefficient, old_z in zip(lam, z_history):
            trial += coefficient * old_z
        x = projection(trial)

    iterations = len(residuals) - 1
    return IterationResult(
        solution=x,
        residual_history=np.asarray(residuals, dtype=float),
        error_history=np.asarray(errors, dtype=float),
        accepted_history=np.array([], dtype=bool),
        max_iterations=max_iterations,
        iterations=iterations,
        converged=converged,
        target_relative_residual=target_relative_residual,
    )


def algorithm2(
    F,
    projection,
    x0,
    tau,
    m,
    nu,
    rho,
    R0,
    xstar,
    max_iterations=10000,
    target_relative_residual=1.0e-8,
):
    """Paper Algorithm 2 with relative fixed-point-residual stopping."""
    if m < 0:
        raise ValueError("m must be nonnegative.")
    if nu <= 0.0:
        raise ValueError("nu must be positive.")
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0, 1).")
    if tau <= 0.0:
        raise ValueError("tau must be positive.")
    if R0 < 0.0:
        raise ValueError("R0 must be nonnegative.")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative.")
    if target_relative_residual < 0.0:
        raise ValueError("target_relative_residual must be nonnegative.")

    x = projection(np.asarray(x0, dtype=float)).copy()
    Rbar = R0 * np.sqrt(1.0 + nu / (1.0 - rho))

    z_history = []
    g_history = []
    residuals = []
    errors = []
    accepted = []
    converged = False

    for k in range(max_iterations + 1):
        z, g, residual = _record_residual(F, projection, x, tau)
        residuals.append(residual)
        errors.append(float(np.linalg.norm(x - xstar)))

        if k == 0:
            threshold = _stopping_threshold(residual, target_relative_residual)

        if residual <= threshold:
            converged = True
            break
        if k == max_iterations:
            break

        z_history.append(z.copy())
        g_history.append(g.copy())
        if len(z_history) > m + 1:
            z_history.pop(0)
            g_history.pop(0)

        lam = anderson_coefficients(g_history)
        trial_argument = np.zeros_like(x)
        for coefficient, old_z in zip(lam, z_history):
            trial_argument += coefficient * old_z
        xhat = projection(trial_argument)

        # Paper (4.1) and Algorithm 2: accept only within the safeguard budget.
        delta = float(np.linalg.norm(xhat - z))
        eta = float(nu * R0**2 * rho**k)
        is_accepted = bool(2.0 * Rbar * delta + delta**2 <= eta)

        accepted.append(is_accepted)
        x = xhat if is_accepted else z

    iterations = len(residuals) - 1
    return IterationResult(
        solution=x,
        residual_history=np.asarray(residuals, dtype=float),
        error_history=np.asarray(errors, dtype=float),
        accepted_history=np.asarray(accepted, dtype=bool),
        max_iterations=max_iterations,
        iterations=iterations,
        converged=converged,
        target_relative_residual=target_relative_residual,
        R0=float(R0),
        Rbar=float(Rbar),
    )
