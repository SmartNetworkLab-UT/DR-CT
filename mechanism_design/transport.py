"""Discrete optimal-transport helpers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def build_cost_matrix(support: np.ndarray, power: float = 1) -> np.ndarray:
    """Build `C[i, j] = |support[i] - support[j]|**power`."""

    support = np.asarray(support, dtype=float)
    return np.abs(support[:, None] - support[None, :]) ** power


def wasserstein_distance_discrete(
    f: np.ndarray,
    fhat: np.ndarray,
    *,
    support: np.ndarray | None = None,
    C: np.ndarray | None = None,
    check_inputs: bool = True,
) -> tuple[float, np.ndarray]:
    """
    Compute the discrete OT distance between two distributions on the same support.

    The LP variables are the coupling `pi[i, j]`.
    """

    f = np.asarray(f, dtype=float)
    fhat = np.asarray(fhat, dtype=float)

    if check_inputs:
        if f.ndim != 1 or fhat.ndim != 1:
            raise ValueError("f and fhat must be one-dimensional arrays.")
        if f.shape != fhat.shape:
            raise ValueError("f and fhat must have the same shape.")
        if np.any(f < -1e-12) or np.any(fhat < -1e-12):
            raise ValueError("Distributions must be nonnegative.")
        if not np.isclose(np.sum(f), 1.0, atol=1e-10):
            raise ValueError(f"f must sum to 1, got {np.sum(f)}.")
        if not np.isclose(np.sum(fhat), 1.0, atol=1e-10):
            raise ValueError(f"fhat must sum to 1, got {np.sum(fhat)}.")

    n = f.size
    if C is None:
        if support is None:
            raise ValueError("Provide either support or C.")
        C = build_cost_matrix(support)
    else:
        C = np.asarray(C, dtype=float)
        if C.shape != (n, n):
            raise ValueError(f"C must have shape ({n}, {n}).")

    c = C.reshape(-1)
    A_eq = []
    b_eq = []

    for i in range(n):
        row = np.zeros(n * n)
        row[i * n : (i + 1) * n] = 1.0
        A_eq.append(row)
        b_eq.append(f[i])

    for j in range(n):
        row = np.zeros(n * n)
        row[j::n] = 1.0
        A_eq.append(row)
        b_eq.append(fhat[j])

    result = linprog(
        c=c,
        A_eq=np.asarray(A_eq),
        b_eq=np.asarray(b_eq),
        bounds=[(0.0, None)] * (n * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"OT LP failed: {result.message}")

    return float(result.fun), result.x.reshape(n, n)
