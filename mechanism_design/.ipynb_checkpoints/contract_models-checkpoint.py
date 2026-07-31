"""Contract models and IC/IR utilities extracted from `contract.ipynb`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, minimize


@dataclass
class ContractSolution:
    """A solved deterministic or robust contract."""

    name: str
    D: np.ndarray
    objective: float
    solver: str
    status: str = ""
    raw_result: object | None = None


@dataclass
class RobustContractSolution:
    """CVXPY robust proposed-contract solution."""

    name: str
    D: np.ndarray | None
    objective: float | None
    solver: str
    status: str
    gamma: float | None = None
    gamma_t: np.ndarray | None = None
    s: np.ndarray | None = None
    s_t: np.ndarray | None = None


@dataclass
class NaiveSolution:
    """Naive scheme derived from the perfect-information contract."""

    D: np.ndarray
    lambdas: np.ndarray
    utility: np.ndarray
    objective: float


@dataclass
class UtilityCheck:
    """Per-contract rewards and incentive diagnostics."""

    lambdas: np.ndarray
    utility_by_true_report_time: np.ndarray
    truthful_by_time: np.ndarray
    utility_total: np.ndarray
    truthful_total: np.ndarray
    ir_ok: bool
    ic_ok: bool
    min_ir: float
    min_ic_gap: float


def normalize_distribution(f: np.ndarray) -> np.ndarray:
    """Return a nonnegative distribution that sums to one."""

    f = np.asarray(f, dtype=float)
    if f.ndim != 1:
        raise ValueError("f must be one-dimensional.")
    if np.any(f < -1e-12):
        raise ValueError("f must be nonnegative.")
    total = float(np.sum(f))
    if total <= 0:
        raise ValueError("f must have positive mass.")
    return f / total


def _pack(D: np.ndarray) -> np.ndarray:
    return np.asarray(D, dtype=float).reshape(-1)


def _unpack(x: np.ndarray, N: int, T: int) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(N, T)


def _daily_cap_constraint(N: int, T: int, Ks: float) -> LinearConstraint:
    rows, cols, data = [], [], []
    for i in range(N):
        for t in range(T):
            rows.append(i)
            cols.append(i * T + t)
            data.append(1.0)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(N, N * T)).tocsr()
    return LinearConstraint(matrix, -np.inf * np.ones(N), np.full(N, Ks, dtype=float))


def _balance_constraint(M: int, f: np.ndarray, Dreq: np.ndarray, N: int, T: int) -> LinearConstraint:
    rows, cols, data = [], [], []
    for t in range(T):
        for i in range(N):
            rows.append(t)
            cols.append(i * T + t)
            data.append(M * f[i])
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(T, N * T)).tocsr()
    return LinearConstraint(matrix, np.asarray(Dreq, dtype=float), np.full(T, np.inf))


def _monotonicity_constraint(N: int, T: int) -> LinearConstraint:
    rows, cols, data, upper = [], [], [], []
    row = 0
    for i in range(N - 1):
        for t in range(T):
            rows += [row, row]
            cols += [(i + 1) * T + t, i * T + t]
            data += [1.0, -1.0]
            upper.append(0.0)
            row += 1
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(row, N * T)).tocsr()
    return LinearConstraint(matrix, -np.inf * np.ones(row), np.asarray(upper))


def _initial_reduction(Dreq: np.ndarray, Krt: np.ndarray, Ks: float, M: int, f: np.ndarray, N: int) -> np.ndarray:
    """Build the same feasible-ish starting point used in the original notebook."""

    D0 = np.tile(Dreq / (M * np.sum(f)), (N, 1))
    D0 = np.minimum(D0, np.tile(Krt, (N, 1)))
    for i in range(N):
        row_sum = float(np.sum(D0[i]))
        if row_sum > Ks + 1e-12:
            D0[i] *= Ks / row_sum
    return D0


def _profit_coefficients(
    *,
    M: int,
    f: np.ndarray,
    pi: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    include_information_rents: bool,
) -> tuple[np.ndarray, np.ndarray]:
    N = f.size
    T = pi.size

    delta_alpha = np.zeros(N)
    delta_beta = np.zeros(N)
    delta_alpha[1:] = alpha[1:] - alpha[:-1]
    delta_beta[1:] = beta[1:] - beta[:-1]

    F_prev = np.zeros(N)
    F_prev[1:] = np.cumsum(f)[:-1]

    quad_obj = np.zeros(N)
    lin_obj = np.zeros((N, T))
    for i in range(N):
        if include_information_rents:
            quad_obj[i] = -M * f[i] * alpha[i] - M * F_prev[i] * delta_alpha[i]
            lin_obj[i] = M * f[i] * (pi - beta[i]) - M * F_prev[i] * delta_beta[i]
        else:
            quad_obj[i] = -M * f[i] * alpha[i]
            lin_obj[i] = M * f[i] * (pi - beta[i])

    return -quad_obj, -lin_obj


def _solve_quadratic_contract(
    *,
    name: str,
    M: int,
    T: int,
    N: int,
    Krt: np.ndarray,
    Ks: float,
    Dreq: np.ndarray,
    pi: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    f: np.ndarray,
    include_information_rents: bool,
    include_monotonicity: bool,
    include_balance: bool,
    verbose: int = 1,
    maxiter: int = 2000,
) -> ContractSolution:
    Krt = np.asarray(Krt, dtype=float).reshape(T)
    Dreq = np.asarray(Dreq, dtype=float).reshape(T)
    pi = np.asarray(pi, dtype=float).reshape(T)
    alpha = np.asarray(alpha, dtype=float).reshape(N)
    beta = np.asarray(beta, dtype=float).reshape(N)
    f = normalize_distribution(f).reshape(N)

    quad_min, lin_min = _profit_coefficients(
        M=M,
        f=f,
        pi=pi,
        alpha=alpha,
        beta=beta,
        include_information_rents=include_information_rents,
    )

    def objective(x: np.ndarray) -> float:
        D = _unpack(x, N, T)
        return float(np.sum(lin_min * D) + np.sum(quad_min[:, None] * D**2))

    def jacobian(x: np.ndarray) -> np.ndarray:
        D = _unpack(x, N, T)
        return _pack(lin_min + 2 * quad_min[:, None] * D)

    def hessian(_: np.ndarray) -> sparse.spmatrix:
        return sparse.diags(np.repeat(2 * quad_min, T), format="csr")

    constraints: list[LinearConstraint] = [_daily_cap_constraint(N, T, Ks)]
    if include_monotonicity:
        constraints.append(_monotonicity_constraint(N, T))
    if include_balance:
        constraints.append(_balance_constraint(M, f, Dreq, N, T))

    bounds = Bounds(np.zeros(N * T), np.tile(Krt, N))
    x0 = _pack(_initial_reduction(Dreq, Krt, Ks, M, f, N))

    result = minimize(
        objective,
        x0,
        method="trust-constr",
        jac=jacobian,
        hess=hessian,
        constraints=constraints,
        bounds=bounds,
        options={"verbose": verbose, "gtol": 1e-9, "xtol": 1e-10, "maxiter": maxiter},
    )

    if not result.success:
        raise RuntimeError(f"{name} optimization failed: {result.message}")

    return ContractSolution(
        name=name,
        D=_unpack(result.x, N, T),
        objective=-float(result.fun),
        solver="scipy.trust-constr",
        status=str(result.message),
        raw_result=result,
    )


def solve_proposed_contract(
    *,
    M: int,
    T: int,
    N: int,
    Krt: np.ndarray,
    Ks: float,
    Dreq: np.ndarray,
    pi: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    f: np.ndarray,
    verbose: int = 1,
) -> ContractSolution:
    """Solve the proposed deterministic contract with IC monotonicity and balance."""

    return _solve_quadratic_contract(
        name="proposed",
        M=M,
        T=T,
        N=N,
        Krt=Krt,
        Ks=Ks,
        Dreq=Dreq,
        pi=pi,
        alpha=alpha,
        beta=beta,
        f=f,
        include_information_rents=True,
        include_monotonicity=True,
        include_balance=True,
        verbose=verbose,
    )


def solve_perfect_information_contract(
    *,
    M: int,
    T: int,
    N: int,
    Krt: np.ndarray,
    Ks: float,
    Dreq: np.ndarray,
    pi: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    f: np.ndarray,
    include_balance: bool = False,
    verbose: int = 1,
) -> ContractSolution:
    """
    Solve the discriminatory/perfect-information benchmark.

    `include_balance=False` preserves the original notebook, where this benchmark
    used the hourly and daily bounds but did not impose the aggregate Dreq balance.
    """

    return _solve_quadratic_contract(
        name="perfect_information",
        M=M,
        T=T,
        N=N,
        Krt=Krt,
        Ks=Ks,
        Dreq=Dreq,
        pi=pi,
        alpha=alpha,
        beta=beta,
        f=f,
        include_information_rents=False,
        include_monotonicity=False,
        include_balance=include_balance,
        verbose=verbose,
    )


def solve_naive_from_perfect(
    *,
    M: int,
    pi: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    perfect_D: np.ndarray,
) -> NaiveSolution:
    """Build the naive scheme from the highest-cost perfect-information contract."""

    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    D = np.asarray(perfect_D, dtype=float)
    N = alpha.size

    D_naive_one_type = D[N - 1]
    lambda_naive = alpha[N - 1] * D_naive_one_type + beta[N - 1]
    D_naive = np.tile(D_naive_one_type, (N, 1))

    utility = (
        D_naive_one_type[None, :] * lambda_naive[None, :]
        - alpha[:, None] * D_naive_one_type[None, :] ** 2
        - beta[:, None] * D_naive_one_type[None, :]
    )
    objective = float(np.sum(M * (pi * D_naive_one_type - lambda_naive * D_naive_one_type)))
    return NaiveSolution(D=D_naive, lambdas=np.tile(lambda_naive, (N, 1)), utility=utility, objective=objective)


def solve_robust_contract_convex(
    *,
    M: int,
    T: int,
    N: int,
    Krt: np.ndarray,
    Ks: float,
    Dreq: np.ndarray,
    pi: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    fhat: np.ndarray,
    r: float,
    c_mat: np.ndarray,
    solver_preference: Iterable[str] = ("MOSEK", "CLARABEL", "ECOS", "SCS"),
    eps_pos: float = 1e-6,
    verbose: bool = False,
) -> RobustContractSolution:
    """Solve the convex robust proposed-contract reformulation with CVXPY."""

    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError("CVXPY is required for solve_robust_contract_convex.") from exc

    Krt = np.asarray(Krt, dtype=float).reshape(T)
    Dreq = np.asarray(Dreq, dtype=float).reshape(T)
    pi = np.asarray(pi, dtype=float).reshape(T)
    alpha = np.asarray(alpha, dtype=float).reshape(N)
    beta = np.asarray(beta, dtype=float).reshape(N)
    fhat = normalize_distribution(fhat).reshape(N)
    c_mat = np.asarray(c_mat, dtype=float).reshape(N, N)

    delta_alpha = np.zeros(N)
    delta_beta = np.zeros(N)
    delta_alpha[1:] = alpha[1:] - alpha[:-1]
    delta_beta[1:] = beta[1:] - beta[:-1]

    D = cp.Variable((N, T))
    gamma = cp.Variable(nonneg=True)
    gamma_t = cp.Variable(T, nonneg=True)
    s = cp.Variable(N)
    s_t = cp.Variable((N, T))

    constraints = [
        D[:-1, :] >= D[1:, :],
        D >= eps_pos,
        D <= Krt.reshape(1, T),
        cp.sum(D, axis=1) <= Ks,
        -gamma_t * r + (fhat @ s_t) >= Dreq,
    ]

    for t in range(T):
        left = cp.vstack([s_t[:, t].T] * N)
        right = M * D[:, t][:, None] + gamma_t[t] * c_mat
        constraints.append(left <= right)

    ell = []
    for i in range(N):
        expr = 0
        for t in range(T):
            if i + 1 <= N - 1:
                tail_sq = cp.sum(cp.multiply(delta_alpha[i + 1 :], cp.square(D[i + 1 :, t])))
                tail_lin = cp.sum(cp.multiply(delta_beta[i + 1 :], D[i + 1 :, t]))
            else:
                tail_sq = 0
                tail_lin = 0
            expr += pi[t] * D[i, t] - alpha[i] * cp.square(D[i, t]) - beta[i] * D[i, t] - tail_sq - tail_lin
        ell.append(M * expr)

    ell = cp.hstack(ell)
    constraints.append(cp.vstack([s.T] * N) <= ell[:, None] + gamma * c_mat)

    problem = cp.Problem(cp.Maximize(-gamma * r + fhat @ s), constraints)

    chosen = next((name for name in solver_preference if name in cp.installed_solvers()), None)
    if chosen is None:
        raise RuntimeError(f"No requested CVXPY solver is installed: {tuple(solver_preference)}.")

    if chosen == "SCS":
        problem.solve(solver=chosen, verbose=verbose, max_iters=200000, eps=1e-4)
    elif chosen == "ECOS":
        problem.solve(solver=chosen, verbose=verbose, max_iters=20000)
    else:
        problem.solve(solver=chosen, verbose=verbose)

    return RobustContractSolution(
        name=f"robust_r_{r:g}",
        D=None if D.value is None else np.asarray(D.value, dtype=float),
        objective=None if problem.value is None else float(problem.value),
        solver=chosen,
        status=str(problem.status),
        gamma=None if gamma.value is None else float(gamma.value),
        gamma_t=None if gamma_t.value is None else np.asarray(gamma_t.value, dtype=float).reshape(T),
        s=None if s.value is None else np.asarray(s.value, dtype=float).reshape(N),
        s_t=None if s_t.value is None else np.asarray(s_t.value, dtype=float),
    )


def compute_contract_utilities(D: np.ndarray, alpha: np.ndarray, beta: np.ndarray, eps: float = 1e-12) -> UtilityCheck:
    """Compute rewards, truthful utilities, IC, and IR for a menu of contracts."""

    D = np.asarray(D, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    N, T = D.shape

    w = np.zeros((N, T))
    for k in range(N - 1):
        w[k] = alpha[k] * (D[k] ** 2 - D[k + 1] ** 2) + beta[k] * (D[k] - D[k + 1])

    base_term = alpha[N - 1] * D[N - 1] ** 2 + beta[N - 1] * D[N - 1]
    lambdas = np.zeros((N, T))
    for i in range(N):
        lambdas[i] = (base_term + np.sum(w[i:], axis=0)) / np.maximum(D[i], eps)

    utility = np.zeros((N, N, T))
    for true_type in range(N):
        for reported_type in range(N):
            utility[true_type, reported_type] = (
                lambdas[reported_type] * D[reported_type]
                - alpha[true_type] * D[reported_type] ** 2
                - beta[true_type] * D[reported_type]
            )

    truthful_by_time = np.array([utility[i, i] for i in range(N)])
    utility_total = np.sum(utility, axis=2)
    truthful_total = np.diag(utility_total)
    ic_gap = truthful_by_time[:, None, :] - utility

    return UtilityCheck(
        lambdas=lambdas,
        utility_by_true_report_time=utility,
        truthful_by_time=truthful_by_time,
        utility_total=utility_total,
        truthful_total=truthful_total,
        ir_ok=bool(np.all(truthful_by_time >= -1e-8)),
        ic_ok=bool(np.all(ic_gap >= -1e-8)),
        min_ir=float(np.min(truthful_by_time)),
        min_ic_gap=float(np.min(ic_gap)),
    )


def compute_discriminatory_utilities(D: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Utility tensor for the perfect-information benchmark."""

    D = np.asarray(D, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    lambdas = alpha[:, None] * D + beta[:, None]
    return D[None, :, :] * lambdas[None, :, :] - alpha[:, None, None] * D[None, :, :] ** 2 - beta[:, None, None] * D[None, :, :]
