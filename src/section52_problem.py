from dataclasses import dataclass
import math
import numpy as np
import scipy.sparse as sp


@dataclass
class Section52Data:
    N: int
    n: int
    l: float
    h: float
    p: float
    q: float
    beta: float
    gamma: float
    A: sp.spmatrix
    B: sp.spmatrix
    AB: sp.spmatrix
    xstar: np.ndarray
    wstar: np.ndarray
    bstar: np.ndarray
    Mq: np.ndarray
    Mp: np.ndarray
    gamma_star: float
    gamma_bar: float
    mu: float
    lambda_min_A: float
    grid: np.ndarray


def theta_power(x, exponent):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.abs(x) ** exponent


def project_box(x, l):
    return np.clip(np.asarray(x, dtype=float), -l, l)


def build_laplacian_2d(N, l):
    h = 2.0 * l / (N + 1)
    off = -np.ones(N - 1, dtype=float)
    main = 2.0 * np.ones(N, dtype=float)
    T = sp.diags([off, main, off], offsets=[-1, 0, 1], format="csr") / h**2
    identity = sp.eye(N, format="csr")
    # Five-point discretization of -Delta on the interior grid.
    A = sp.kron(identity, T, format="csr") + sp.kron(T, identity, format="csr")
    return A, h


def build_skew_convection_2d(N, l, beta):
    h = 2.0 * l / (N + 1)
    off = np.ones(N - 1, dtype=float)
    D = sp.diags([-off, off], offsets=[-1, 1], format="csr") / (2.0 * h)
    identity = sp.eye(N, format="csr")
    # The centered-difference convection matrix is skew-symmetric.
    return (beta * (sp.kron(identity, D, format="csr") + sp.kron(D, identity, format="csr"))).tocsr()


def smooth_manufactured_solution(N, l, amplitude=0.35):
    h = 2.0 * l / (N + 1)
    grid = np.linspace(-l + h, l - h, N)
    xstar = np.zeros(N * N, dtype=float)
    for j in range(N):
        for i in range(N):
            index = i + j * N
            xstar[index] = amplitude * math.sin(math.pi * grid[i] / l) * math.sin(math.pi * grid[j] / l)
    return xstar, grid


def make_boundary_manufactured_solution(N, l, amplitude=0.35, normal_magnitude=0.5):
    """Put the first grid column at +l and the last grid column at -l."""
    xstar, grid = smooth_manufactured_solution(N, l, amplitude=amplitude)
    wstar = np.zeros(N * N, dtype=float)
    for j in range(N):
        plus_index = j * N
        minus_index = (N - 1) + j * N
        xstar[plus_index] = l
        xstar[minus_index] = -l
        wstar[plus_index] = normal_magnitude
        wstar[minus_index] = -normal_magnitude
    return xstar, wstar, grid


def _secant_slopes(xstar, l, exponent):
    values = np.empty_like(xstar, dtype=float)
    tolerance = 100.0 * np.finfo(float).eps * max(1.0, abs(l))
    for i, value in enumerate(xstar):
        a = abs(value)
        if abs(a - l) <= tolerance:
            values[i] = exponent * l ** (exponent - 1.0)
        else:
            values[i] = (l**exponent - a**exponent) / (l - a)
    return values


def _laplacian_extreme_eigenvalues(N, h):
    angle = math.pi / (2.0 * (N + 1))
    lambda_min = 8.0 * math.sin(angle) ** 2 / h**2
    lambda_max = 8.0 * math.cos(angle) ** 2 / h**2
    return lambda_min, lambda_max


def _gamma_star(A, Mq, Mp, lambda_min_A):
    if np.max(Mp) < lambda_min_A:
        return 0.0
    Dinv = np.diag(1.0 / np.sqrt(Mq))
    H = Dinv @ (np.diag(Mp) - A.toarray()) @ Dinv
    H = 0.5 * (H + H.T)
    return max(0.0, float(np.linalg.eigvalsh(H)[-1]))


def _mu(A, Mq, Mp, gamma):
    H = A.toarray() + np.diag(gamma * Mq - Mp)
    H = 0.5 * (H + H.T)
    return float(np.linalg.eigvalsh(H)[0])


def build_problem(
    N=15,
    l=1.0,
    p=6.0,
    q=0.75,
    beta=0.15,
    amplitude=0.35,
    gamma=1.0,
    xstar=None,
    wstar=None,
):
    A, h = build_laplacian_2d(N, l)
    B = build_skew_convection_2d(N, l, beta)
    AB = (A + B).tocsr()
    n = N * N

    if xstar is None:
        xstar, grid = smooth_manufactured_solution(N, l, amplitude=amplitude)
    else:
        xstar = np.asarray(xstar, dtype=float).copy()
        grid = np.linspace(-l + h, l - h, N)
    if wstar is None:
        wstar = np.zeros(n, dtype=float)
    else:
        wstar = np.asarray(wstar, dtype=float).copy()

    Mq = _secant_slopes(xstar, l, q)
    Mp = _secant_slopes(xstar, l, p)
    lambda_min_A, _ = _laplacian_extreme_eigenvalues(N, h)
    gamma_star = _gamma_star(A, Mq, Mp, lambda_min_A)
    gamma_bar = (p * l ** (p - 1.0) - lambda_min_A) / (q * l ** (q - 1.0))
    if gamma <= gamma_star:
        raise ValueError("gamma must be larger than gamma_star.")
    mu = _mu(A, Mq, Mp, gamma)
    if mu <= 0.0:
        raise ValueError("mu must be positive.")

    # The manufactured forcing ensures F(xstar) = -wstar.
    bstar = AB.dot(xstar) + gamma * theta_power(xstar, q) - theta_power(xstar, p) + wstar

    return Section52Data(
        N=N,
        n=n,
        l=l,
        h=h,
        p=p,
        q=q,
        beta=beta,
        gamma=gamma,
        A=A,
        B=B,
        AB=AB,
        xstar=xstar,
        wstar=wstar,
        bstar=bstar,
        Mq=Mq,
        Mp=Mp,
        gamma_star=gamma_star,
        gamma_bar=gamma_bar,
        mu=mu,
        lambda_min_A=lambda_min_A,
        grid=grid,
    )


def F52(x, data):
    return data.AB.dot(x) + data.gamma * theta_power(x, data.q) - theta_power(x, data.p) - data.bstar



def R0_section52(data, x0):
    """Radius bound (2.5) for sigma = 1."""
    return float(np.linalg.norm(F52(x0, data)) / data.mu)
