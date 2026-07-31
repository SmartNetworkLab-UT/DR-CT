"""Direct nonlinear robust optimization with  CasADi."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


from .contract_models import normalize_distribution


@dataclass(frozen=True)
class NonconvexLayout:
    """Flat-vector layout for the nonlinear robust model."""

    T: int
    N: int

    @property
    def n_mu(self) -> int:
        return self.T * self.N

    @property
    def n_lam(self) -> int:
        return self.T * self.N

    @property
    def n_xi(self) -> int:
        return self.T * self.N

    @property
    def n_omega(self) -> int:
        return self.N

    @property
    def n_x(self) -> int:
        return self.T

    @property
    def n_z(self) -> int:
        return self.N * self.N

    @property
    def n_y(self) -> int:
        return self.T * self.N * self.N

    @property
    def n_vars(self) -> int:
        return self.n_mu + self.n_lam + self.n_xi + self.n_omega + self.n_x + self.n_z + self.n_y

    @property
    def offsets(self) -> dict[str, int]:
        off_mu = 0
        off_lam = off_mu + self.n_mu
        off_xi = off_lam + self.n_lam
        off_omega = off_xi + self.n_xi
        off_x = off_omega + self.n_omega
        off_z = off_x + self.n_x
        off_y = off_z + self.n_z
        return {
            "mu": off_mu,
            "lam": off_lam,
            "xi": off_xi,
            "omega": off_omega,
            "x": off_x,
            "z": off_z,
            "y": off_y,
        }

    def pack(
        self,
        mu: np.ndarray,
        lam: np.ndarray,
        xi: np.ndarray,
        omega: np.ndarray,
        x: np.ndarray,
        z: np.ndarray,
        y: np.ndarray,
        *,
        order: str = "C",
    ) -> np.ndarray:
        if order == "F":
            y_parts = [np.asarray(y[t]).reshape(-1, order="F") for t in range(self.T)]
            return np.concatenate(
                [
                    np.asarray(mu).reshape(-1, order="F"),
                    np.asarray(lam).reshape(-1, order="F"),
                    np.asarray(xi).reshape(-1, order="F"),
                    np.asarray(omega).reshape(-1),
                    np.asarray(x).reshape(-1),
                    np.asarray(z).reshape(-1, order="F"),
                    *y_parts,
                ]
            )
        return np.concatenate(
            [
                np.asarray(mu).ravel(),
                np.asarray(lam).ravel(),
                np.asarray(xi).ravel(),
                np.asarray(omega).ravel(),
                np.asarray(x).ravel(),
                np.asarray(z).ravel(),
                np.asarray(y).ravel(),
            ]
        )

    def unpack(self, v: np.ndarray, *, order: str = "C") -> tuple[np.ndarray, ...]:
        v = np.asarray(v, dtype=float).reshape(-1)
        offsets = self.offsets
        if order == "F":
            k = offsets["mu"]
            mu = v[k : k + self.n_mu].reshape(self.T, self.N, order="F")
            k = offsets["lam"]
            lam = v[k : k + self.n_lam].reshape(self.T, self.N, order="F")
            k = offsets["xi"]
            xi = v[k : k + self.n_xi].reshape(self.T, self.N, order="F")
            k = offsets["omega"]
            omega = v[k : k + self.n_omega]
            k = offsets["x"]
            x = v[k : k + self.n_x]
            k = offsets["z"]
            z = v[k : k + self.n_z].reshape(self.N, self.N, order="F")
            k = offsets["y"]
            y = np.zeros((self.T, self.N, self.N))
            for t in range(self.T):
                y[t] = v[k : k + self.N * self.N].reshape(self.N, self.N, order="F")
                k += self.N * self.N
            return mu, lam, xi, omega, x, z, y

        k = offsets["mu"]
        mu = v[k : k + self.n_mu].reshape(self.T, self.N)
        k = offsets["lam"]
        lam = v[k : k + self.n_lam].reshape(self.T, self.N)
        k = offsets["xi"]
        xi = v[k : k + self.n_xi].reshape(self.T, self.N)
        k = offsets["omega"]
        omega = v[k : k + self.n_omega]
        k = offsets["x"]
        x = v[k : k + self.n_x]
        k = offsets["z"]
        z = v[k : k + self.n_z].reshape(self.N, self.N)
        k = offsets["y"]
        y = v[k : k + self.n_y].reshape(self.T, self.N, self.N)
        return mu, lam, xi, omega, x, z, y


@dataclass
class NonconvexResult:
    """Result returned by the direct robust nonlinear optimizer."""

    solver: str
    success: bool
    status: str
    message: str
    objective: float
    D: np.ndarray
    variables: dict[str, np.ndarray]
    Zi: np.ndarray
    x: np.ndarray
    vector: np.ndarray
    raw_result: Any | None = None


def build_type_distance_matrix(N: int) -> np.ndarray:
    """Default ground-cost matrix `|i-j|` for type-index transport."""

    idx = np.arange(N)
    return np.abs(idx[:, None] - idx[None, :]).astype(float)


def initial_nonconvex_point(layout: NonconvexLayout, fhat: np.ndarray, x_value: float = 0.1) -> tuple[np.ndarray, ...]:
    """Construct the diagonal feasible initialization used by the original notebook."""

    T, N = layout.T, layout.N
    x = np.full(T, x_value, dtype=float)
    z = np.zeros((N, N), dtype=float)
    np.fill_diagonal(z, fhat)

    y = np.zeros((T, N, N), dtype=float)
    for t in range(T):
        np.fill_diagonal(y[t], x[t] * fhat)

    return (
        np.zeros((T, N)),
        np.zeros((T, N)),
        np.zeros((T, N)),
        np.zeros(N),
        x,
        z,
        y,
    )


def check_nonconvex_initialization(
    *,
    layout: NonconvexLayout,
    fhat: np.ndarray,
    C: np.ndarray,
    r: float,
    vector: np.ndarray,
    order: str = "C",
) -> dict[str, float]:
    """Return the equality and transport-budget residuals of an initial point."""

    _, _, _, _, x, z, y = layout.unpack(vector, order=order)
    eq_z = float(np.max(np.abs(z.sum(axis=0) - fhat)))
    eq_y = 0.0
    for t in range(layout.T):
        eq_y = max(eq_y, float(np.max(np.abs(y[t].sum(axis=0) - x[t] * fhat))))
    return {
        "z_column_residual": eq_z,
        "y_column_residual": eq_y,
        "z_transport_cost": float(np.sum(z * C)),
        "max_y_transport_slack": float(max(np.sum(y[t] * C) - x[t] * r for t in range(layout.T))),
    }


def _compute_ZY(z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return z.sum(axis=1), y.sum(axis=2)


def _objective_numpy(
    v: np.ndarray,
    *,
    layout: NonconvexLayout,
    M: int,
    Krt: np.ndarray,
    Ks: float,
    Dreq: np.ndarray,
    pi: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    eps_z: float,
    reg: float,
) -> float:
    mu, lam, xi, omega, x, z, y = layout.unpack(v, order="C")
    Zi, Yi = _compute_ZY(z, y)

    Zi_safe = Zi + eps_z
    S = np.concatenate(([0.0], np.cumsum(Zi_safe[:-1])))
    alpha_prev = np.concatenate(([alpha[0]], alpha[:-1]))
    beta_prev = np.concatenate(([beta[0]], beta[:-1]))

    lin = float(np.sum(xi * Krt[:, None]) + Ks * np.sum(omega) - np.dot(x, Dreq))
    Ai = Zi_safe * alpha + (alpha - alpha_prev) * S
    Ai_safe = Ai + 1e-12

    mu_prev = np.zeros_like(mu)
    mu_prev[:, 1:] = mu[:, :-1]
    dmu = mu - mu_prev

    main = Zi_safe[None, :] * pi[:, None] - Zi_safe[None, :] * beta[None, :] - (beta - beta_prev)[None, :] * S[None, :]
    Bi = M * main + dmu + lam - xi - omega[None, :] + M * Yi
    frac = np.sum((Bi**2) / (4.0 * M * Ai_safe[None, :]))
    value = lin + frac
    if reg > 0:
        value += reg * float(np.dot(v, v))
    return float(value)


def _linear_constraints(layout: NonconvexLayout, fhat: np.ndarray, C: np.ndarray, r: float) -> list[LinearConstraint]:
    T, N = layout.T, layout.N
    offsets = layout.offsets

    rows, cols, data = [], [], []
    for j in range(N):
        for i in range(N):
            rows.append(j)
            cols.append(offsets["z"] + i * N + j)
            data.append(1.0)
    A_eq_z = sparse.coo_matrix((data, (rows, cols)), shape=(N, layout.n_vars)).tocsr()

    rows, cols, data = [], [], []
    row = 0
    for t in range(T):
        for j in range(N):
            for i in range(N):
                y_flat = (t * N + i) * N + j
                rows.append(row)
                cols.append(offsets["y"] + y_flat)
                data.append(1.0)
            rows.append(row)
            cols.append(offsets["x"] + t)
            data.append(-fhat[j])
            row += 1
    A_eq_y = sparse.coo_matrix((data, (rows, cols)), shape=(T * N, layout.n_vars)).tocsr()
    A_eq = sparse.vstack([A_eq_z, A_eq_y], format="csr")
    b_eq = np.concatenate([fhat, np.zeros(T * N)])

    rows, cols, data = [], [], []
    for i in range(N):
        for j in range(N):
            rows.append(0)
            cols.append(offsets["z"] + i * N + j)
            data.append(C[i, j])
    A_z = sparse.coo_matrix((data, (rows, cols)), shape=(1, layout.n_vars)).tocsr()

    rows, cols, data = [], [], []
    for t in range(T):
        for i in range(N):
            for j in range(N):
                y_flat = (t * N + i) * N + j
                rows.append(t)
                cols.append(offsets["y"] + y_flat)
                data.append(C[i, j])
        rows.append(t)
        cols.append(offsets["x"] + t)
        data.append(-r)
    A_y = sparse.coo_matrix((data, (rows, cols)), shape=(T, layout.n_vars)).tocsr()

    return [
        LinearConstraint(A_eq, b_eq, b_eq),
        LinearConstraint(A_z, -np.inf, np.array([r])),
        LinearConstraint(A_y, -np.inf * np.ones(T), np.zeros(T)),
    ]


def compute_nonconvex_reduction(
    *,
    M: int,
    Krt: np.ndarray,
    pi: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    mu: np.ndarray,
    lam: np.ndarray,
    xi: np.ndarray,
    omega: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    eps_z: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover `D_star` from the nonlinear model variables."""

    Zi, Yi = _compute_ZY(z, y)
    Zi_safe = Zi + eps_z
    S = np.concatenate(([0.0], np.cumsum(Zi_safe[:-1])))
    alpha_prev = np.concatenate(([alpha[0]], alpha[:-1]))
    beta_prev = np.concatenate(([beta[0]], beta[:-1]))

    Ai = Zi_safe * alpha + (alpha - alpha_prev) * S
    Ai_safe = Ai + 1e-12

    mu_prev = np.zeros_like(mu)
    mu_prev[:, 1:] = mu[:, :-1]
    dmu = mu - mu_prev

    main = Zi_safe[None, :] * pi[:, None] - Zi_safe[None, :] * beta[None, :] - (beta - beta_prev)[None, :] * S[None, :]
    numerator = M * main + dmu + lam - xi - omega[None, :] + M * Yi
    D_star = numerator / (2.0 * M * Ai_safe[None, :])
    D_star = np.maximum(D_star, 0.0)
    D_star = np.minimum(D_star, Krt[:, None])
    return D_star, Zi, Yi


def solve_nonconvex_robust(
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
    r: float = 0.1,
    fhat: np.ndarray | None = None,
    C: np.ndarray | None = None,
    solver: str = "casadi",
    eps_z: float = 1e-9,
    reg: float = 1e-8,
    x0_value: float = 0.1,
    verbose: bool = False,
    casadi_options: dict[str, Any] | None = None,
) -> NonconvexResult:
    """
    Solve the direct nonlinear robust model.

    Parameters
    ----------
    solver:
        `"casadi"` uses CasADi/IPOPT and is the default. 
    """

    layout = NonconvexLayout(T=T, N=N)
    Krt = np.asarray(Krt, dtype=float).reshape(T)
    Dreq = np.asarray(Dreq, dtype=float).reshape(T)
    pi = np.asarray(pi, dtype=float).reshape(T)
    alpha = np.asarray(alpha, dtype=float).reshape(N)
    beta = np.asarray(beta, dtype=float).reshape(N)
    fhat = normalize_distribution(np.ones(N) / N if fhat is None else fhat)
    C = build_type_distance_matrix(N) if C is None else np.asarray(C, dtype=float).reshape(N, N)

    solver = solver.lower()
   
    if solver == "casadi":
        return _solve_with_casadi(
            layout=layout,
            M=M,
            Krt=Krt,
            Ks=Ks,
            Dreq=Dreq,
            pi=pi,
            alpha=alpha,
            beta=beta,
            r=r,
            fhat=fhat,
            C=C,
            eps_z=eps_z,
            reg=reg,
            x0_value=x0_value,
            verbose=verbose,
            options=casadi_options,
        )
    raise ValueError("solver must be 'casadi'.")


def _merge_casadi_options(default_options: dict[str, Any], options: dict[str, Any] | None) -> dict[str, Any]:
    if not options:
        return default_options
    merged = dict(default_options)
    for key, value in options.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged



def _solve_with_casadi(
    *,
    layout: NonconvexLayout,
    M: int,
    Krt: np.ndarray,
    Ks: float,
    Dreq: np.ndarray,
    pi: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    r: float,
    fhat: np.ndarray,
    C: np.ndarray,
    eps_z: float,
    reg: float,
    x0_value: float,
    verbose: bool,
    options: dict[str, Any] | None,
) -> NonconvexResult:
    try:
        import casadi as ca
    except ImportError as exc:
        raise ImportError("CasADi is not installed. Install `casadi` .") from exc

    T, N = layout.T, layout.N
    mu = ca.MX.sym("mu", T, N)
    lam = ca.MX.sym("lam", T, N)
    xi = ca.MX.sym("xi", T, N)
    omega = ca.MX.sym("omega", N)
    x = ca.MX.sym("x", T)
    z = ca.MX.sym("z", N, N)
    y = [ca.MX.sym(f"y_{t}", N, N) for t in range(T)]

    def vec(expr):
        return ca.reshape(expr, expr.numel(), 1)

    decision_vector = ca.vertcat(vec(mu), vec(lam), vec(xi), vec(omega), vec(x), vec(z), *[vec(y_t) for y_t in y])

    Zi = [sum(z[i, j] for j in range(N)) for i in range(N)]
    Zi_safe = [Zi[i] + eps_z for i in range(N)]
    S = []
    for i in range(N):
        S.append(0 if i == 0 else sum(Zi_safe[p] for p in range(i)))

    alpha_prev = np.concatenate(([alpha[0]], alpha[:-1]))
    beta_prev = np.concatenate(([beta[0]], beta[:-1]))

    objective = sum(xi[t, i] * Krt[t] for t in range(T) for i in range(N))
    objective += Ks * sum(omega[i] for i in range(N))
    objective -= sum(x[t] * Dreq[t] for t in range(T))

    for t in range(T):
        for i in range(N):
            Ai = Zi_safe[i] * alpha[i] + (alpha[i] - alpha_prev[i]) * S[i] + 1e-12
            mu_prev = 0 if i == 0 else mu[t, i - 1]
            dmu = mu[t, i] - mu_prev
            Yi = sum(y[t][i, j] for j in range(N))
            main = Zi_safe[i] * pi[t] - Zi_safe[i] * beta[i] - (beta[i] - beta_prev[i]) * S[i]
            Bi = M * main + dmu + lam[t, i] - xi[t, i] - omega[i] + M * Yi
            objective += (Bi**2) / (4.0 * M * Ai)

    if reg > 0:
        objective += reg * ca.dot(decision_vector, decision_vector)

    constraints = []
    lbg = []
    ubg = []

    for j in range(N):
        constraints.append(sum(z[i, j] for i in range(N)) - fhat[j])
        lbg.append(0.0)
        ubg.append(0.0)

    for t in range(T):
        for j in range(N):
            constraints.append(sum(y[t][i, j] for i in range(N)) - x[t] * fhat[j])
            lbg.append(0.0)
            ubg.append(0.0)

    constraints.append(sum(C[i, j] * z[i, j] for i in range(N) for j in range(N)))
    lbg.append(-np.inf)
    ubg.append(r)

    for t in range(T):
        constraints.append(sum(C[i, j] * y[t][i, j] for i in range(N) for j in range(N)) - x[t] * r)
        lbg.append(-np.inf)
        ubg.append(0.0)

    nlp = {"x": decision_vector, "f": objective, "g": ca.vertcat(*constraints)}
    default_options = {
        "print_time": verbose,
        "ipopt": {
            "print_level": 5 if verbose else 0,
            "max_iter": 2000,
            "tol": 1e-8,
            "constr_viol_tol": 1e-8,
        },
    }
    solver = ca.nlpsol("nonconvex_robust_solver", "ipopt", nlp, _merge_casadi_options(default_options, options))

    initial = initial_nonconvex_point(layout, fhat, x_value=x0_value)
    v0 = layout.pack(*initial, order="F")
    solution = solver(
        x0=v0,
        lbx=np.zeros(layout.n_vars),
        ubx=np.full(layout.n_vars, np.inf),
        lbg=np.asarray(lbg, dtype=float),
        ubg=np.asarray(ubg, dtype=float),
    )

    vector = np.asarray(solution["x"], dtype=float).reshape(-1)
    mu_v, lam_v, xi_v, omega_v, x_v, z_v, y_v = layout.unpack(vector, order="F")
    D_star, Zi_star, _ = compute_nonconvex_reduction(
        M=M,
        Krt=Krt,
        pi=pi,
        alpha=alpha,
        beta=beta,
        mu=mu_v,
        lam=lam_v,
        xi=xi_v,
        omega=omega_v,
        z=z_v,
        y=y_v,
        eps_z=eps_z,
    )
    stats = solver.stats()
    status = str(stats.get("return_status", ""))

    return NonconvexResult(
        solver="casadi.ipopt",
        success=bool(stats.get("success", False)),
        status=status,
        message=status,
        objective=float(solution["f"]),
        D=D_star,
        variables={"mu": mu_v, "lam": lam_v, "xi": xi_v, "omega": omega_v, "x": x_v, "z": z_v, "y": y_v},
        Zi=Zi_star,
        x=x_v,
        vector=vector,
        raw_result=solution,
    )
