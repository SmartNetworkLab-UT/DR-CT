"""Parvizi-paper figure templates adapted to the Python simulation code.

The original baseline code mixed MATLAB scripts, saved ``.mat`` files, and
notebook plotting cells.  This module keeps the old figure families while
feeding them from the current ``mechanism_design`` scenario and solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .contract_models import (
    compute_contract_utilities,
    normalize_distribution,
    solve_naive_from_perfect,
    solve_perfect_information_contract,
    solve_proposed_contract,
    solve_robust_contract_convex,
)
from .data import build_load_scenario
from .plots import DEFAULT_FIGURE_DPI, plot_ic_curves, save_figure
from .transport import build_cost_matrix


SCHEME_ORDER = ("proposed", "discriminatory", "naive", "stackelberg", "linear")

SCHEME_LABELS = {
    "proposed": "proposed load reduction scheme",
    "discriminatory": "discriminatory load reduction scheme",
    "naive": "load reduction scheme without type verification",
    "stackelberg": "Stackelberg load reduction scheme",
    "linear": "Linear contract load reduction scheme",
}

SCHEME_COLORS = {
    "proposed": "red",
    "discriminatory": "green",
    "naive": "blue",
    "stackelberg": "gold",
    "linear": "purple",
}

SCHEME_MARKERS = {
    "proposed": "o-",
    "discriminatory": "*-",
    "naive": "^-",
    "stackelberg": "s-",
    "linear": "p-",
}

ROBUST_COLORS = ("black", "tab:cyan", "tab:brown", "tab:pink", "tab:olive")
ROBUST_MARKERS = ("D-", "X-", "v-", "<-", ">-")


@dataclass
class SchemeResult:
    """One baseline scheme in the adapted Parvizi figure suite."""

    key: str
    display_name: str
    D: np.ndarray
    lambdas: np.ndarray


@dataclass
class ScaleExperimentResult:
    """Aggregated metrics for scaling figures."""

    x_values: np.ndarray
    operator_utility: dict[str, np.ndarray]
    customer_utility: dict[str, np.ndarray]
    social_welfare: dict[str, np.ndarray]


@dataclass
class SensitivityResult:
    """Demand-prediction sensitivity data."""

    percent_changes: np.ndarray
    operator_error_percent: np.ndarray
    demand_reduction_error_percent: np.ndarray


def _configure_style() -> None:
    """Apply a compact publication style similar to the old notebooks."""

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
        }
    )


def _close_all_figures() -> None:
    """Release matplotlib figures opened by batch-generation helpers."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.close("all")


def _as_matrix(values: np.ndarray, N: int, T: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (N, T):
        raise ValueError(f"Expected shape {(N, T)}, got {values.shape}.")
    return values


def robust_scheme_key(radius: float) -> str:
    """Return a stable dictionary key for a robust radius."""

    return f"robust_r_{float(radius):g}".replace(".", "p").replace("-", "m")


def robust_scheme_label(radius: float) -> str:
    """Display label for a robust proposed contract."""

    return f"robust proposed contract, r={float(radius):g}"


def _scheme_color(key: str, index: int = 0) -> str:
    if key in SCHEME_COLORS:
        return SCHEME_COLORS[key]
    return ROBUST_COLORS[index % len(ROBUST_COLORS)]


def _scheme_marker(key: str, index: int = 0) -> str:
    if key in SCHEME_MARKERS:
        return SCHEME_MARKERS[key]
    return ROBUST_MARKERS[index % len(ROBUST_MARKERS)]


def _scheme_order_keys(schemes: dict[str, object]) -> list[str]:
    ordered = [key for key in SCHEME_ORDER if key in schemes]
    ordered.extend(key for key in schemes if key not in ordered)
    return ordered


def customer_utility_matrix(D: np.ndarray, lambdas: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return per-type, per-hour customer utility."""

    D = np.asarray(D, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    return lambdas * D - alpha[:, None] * D**2 - beta[:, None] * D


def customer_utility_by_type(scheme: SchemeResult, scenario) -> np.ndarray:
    """Individual daily customer utility by type."""

    return np.sum(customer_utility_matrix(scheme.D, scheme.lambdas, scenario.alpha, scenario.beta), axis=1)


def total_customer_utility(scheme: SchemeResult, scenario, f: np.ndarray) -> float:
    """Distribution-weighted total customer utility across all customers."""

    f = normalize_distribution(f)
    utility = customer_utility_matrix(scheme.D, scheme.lambdas, scenario.alpha, scenario.beta)
    return float(np.sum(scenario.M * f[:, None] * utility))


def operator_utility(scheme: SchemeResult, scenario, f: np.ndarray) -> float:
    """Central/grid operator utility for a scheme."""

    f = normalize_distribution(f)
    return float(np.sum(scenario.M * f[:, None] * (scenario.pi[None, :] * scheme.D - scheme.lambdas * scheme.D)))


def _discriminatory_lambdas(D: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return alpha[:, None] * D + beta[:, None]


def _linear_contract(scenario) -> SchemeResult:
    D = np.tile(scenario.Dreq / scenario.M, (scenario.N, 1))
    D = np.minimum(D, scenario.Krt.reshape(1, -1))

    daily = np.sum(D, axis=1)
    too_large = daily > scenario.Ks
    if np.any(too_large):
        D[too_large] *= (scenario.Ks / daily[too_large])[:, None]

    lambdas = scenario.alpha.max() * D + scenario.beta.max()
    return SchemeResult("linear", SCHEME_LABELS["linear"], D, lambdas)


def _hourly_stackelberg_start(target: float, M: int, f: np.ndarray, upper: float) -> np.ndarray:
    x0 = np.full(f.size, target / M, dtype=float)
    x0 = np.minimum(x0, upper)
    current = M * float(np.dot(f, x0))
    if current <= 0:
        return x0
    return np.minimum(x0 * (target / current), upper)


def solve_stackelberg_contract(scenario, f: np.ndarray) -> SchemeResult:
    """
    Solve the Stackelberg benchmark used by the old figures.

    The old MATLAB code solved one hourly quadratic program in rewards.  This
    version writes the same follower-response problem in demand reductions:
    each type's reward is the inverse best response
    ``lambda_it = 2 alpha_i D_it + beta_i``.
    """

    f = normalize_distribution(f)
    N, T = scenario.N, scenario.T
    D = np.zeros((N, T), dtype=float)

    for t in range(T):
        pi_t = float(scenario.pi[t])
        target = float(scenario.Dreq[t])
        upper = np.full(N, float(scenario.Krt[t]), dtype=float)

        def objective(x: np.ndarray) -> float:
            return float(np.sum(scenario.M * f * (2.0 * scenario.alpha * x**2 - (pi_t - scenario.beta) * x)))

        def jacobian(x: np.ndarray) -> np.ndarray:
            return scenario.M * f * (4.0 * scenario.alpha * x - (pi_t - scenario.beta))

        constraint = LinearConstraint((scenario.M * f).reshape(1, -1), [target], [target])
        x0 = _hourly_stackelberg_start(target, scenario.M, f, float(scenario.Krt[t]))
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            jac=jacobian,
            bounds=Bounds(np.zeros(N), upper),
            constraints=[constraint],
            options={"ftol": 1e-10, "maxiter": 500, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"Stackelberg hourly solve failed at hour {t + 1}: {result.message}")
        D[:, t] = result.x

    lambdas = 2.0 * scenario.alpha[:, None] * D + scenario.beta[:, None]
    return SchemeResult("stackelberg", SCHEME_LABELS["stackelberg"], D, lambdas)


def build_baseline_schemes(
    scenario,
    f: np.ndarray,
    *,
    verbose: int = 0,
    discriminatory_balance: bool = True,
) -> dict[str, SchemeResult]:
    """Build the five baseline schemes used by the Parvizi-paper figures."""

    f = normalize_distribution(f)
    proposed = solve_proposed_contract(
        M=scenario.M,
        T=scenario.T,
        N=scenario.N,
        Krt=scenario.Krt,
        Ks=scenario.Ks,
        Dreq=scenario.Dreq,
        pi=scenario.pi,
        alpha=scenario.alpha,
        beta=scenario.beta,
        f=f,
        verbose=verbose,
    )
    proposed_check = compute_contract_utilities(proposed.D, scenario.alpha, scenario.beta)

    discriminatory = solve_perfect_information_contract(
        M=scenario.M,
        T=scenario.T,
        N=scenario.N,
        Krt=scenario.Krt,
        Ks=scenario.Ks,
        Dreq=scenario.Dreq,
        pi=scenario.pi,
        alpha=scenario.alpha,
        beta=scenario.beta,
        f=f,
        include_balance=discriminatory_balance,
        verbose=verbose,
    )
    discriminatory_lambdas = _discriminatory_lambdas(discriminatory.D, scenario.alpha, scenario.beta)

    naive = solve_naive_from_perfect(
        M=scenario.M,
        pi=scenario.pi,
        alpha=scenario.alpha,
        beta=scenario.beta,
        perfect_D=discriminatory.D,
    )

    schemes = {
        "proposed": SchemeResult("proposed", SCHEME_LABELS["proposed"], proposed.D, proposed_check.lambdas),
        "discriminatory": SchemeResult(
            "discriminatory",
            SCHEME_LABELS["discriminatory"],
            discriminatory.D,
            discriminatory_lambdas,
        ),
        "naive": SchemeResult("naive", SCHEME_LABELS["naive"], naive.D, naive.lambdas),
        "stackelberg": solve_stackelberg_contract(scenario, f),
        "linear": _linear_contract(scenario),
    }

    for scheme in schemes.values():
        _as_matrix(scheme.D, scenario.N, scenario.T)
        _as_matrix(scheme.lambdas, scenario.N, scenario.T)
    return schemes


def build_robust_schemes(
    scenario,
    f: np.ndarray,
    *,
    robust_radii: tuple[float, ...] = (0.1, 0.8),
    c_mat: np.ndarray | None = None,
    solver_preference: tuple[str, ...] = ("MOSEK", "CLARABEL", "ECOS", "SCS"),
    verbose: bool = False,
) -> dict[str, SchemeResult]:
    """Build robust proposed contracts using the same convex model as `contract.ipynb`."""

    f = normalize_distribution(f)
    if c_mat is None:
        c_mat = build_cost_matrix(np.arange(scenario.N))

    schemes: dict[str, SchemeResult] = {}
    for radius in robust_radii:
        solution = solve_robust_contract_convex(
            M=scenario.M,
            T=scenario.T,
            N=scenario.N,
            Krt=scenario.Krt,
            Ks=scenario.Ks,
            Dreq=scenario.Dreq,
            pi=scenario.pi,
            alpha=scenario.alpha,
            beta=scenario.beta,
            fhat=f,
            r=float(radius),
            c_mat=c_mat,
            solver_preference=solver_preference,
            verbose=verbose,
        )
        if solution.D is None:
            raise RuntimeError(f"Robust contract r={radius:g} did not return a reduction matrix. Status: {solution.status}")

        check = compute_contract_utilities(solution.D, scenario.alpha, scenario.beta)
        key = robust_scheme_key(float(radius))
        schemes[key] = SchemeResult(
            key=key,
            display_name=robust_scheme_label(float(radius)),
            D=solution.D,
            lambdas=check.lambdas,
        )

    return schemes


def build_baseline_and_robust_schemes(
    scenario,
    f: np.ndarray,
    *,
    robust_radii: tuple[float, ...] = (0.1, 0.8),
    verbose: int = 0,
    robust_verbose: bool = False,
    discriminatory_balance: bool = True,
) -> dict[str, SchemeResult]:
    """Build Parvizi baselines plus robust proposed contracts."""

    schemes = build_baseline_schemes(
        scenario,
        f,
        verbose=verbose,
        discriminatory_balance=discriminatory_balance,
    )
    schemes.update(
        build_robust_schemes(
            scenario,
            f,
            robust_radii=robust_radii,
            verbose=robust_verbose,
        )
    )
    return schemes


def _ordered_schemes(schemes: dict[str, SchemeResult]) -> list[SchemeResult]:
    return [schemes[key] for key in _scheme_order_keys(schemes)]


def plot_customer_utilities(
    schemes: dict[str, SchemeResult],
    scenario,
    *,
    save_path: str | Path | None = None,
    dpi: int = DEFAULT_FIGURE_DPI,
):
    """Grouped bar chart matching ``CUs_utility.pdf``."""

    import matplotlib.pyplot as plt

    _configure_style()
    types = scenario.alpha
    x = np.arange(scenario.N)
    ordered = _ordered_schemes(schemes)
    width = min(0.8 / max(len(ordered), 1), 0.15)
    center = (len(ordered) - 1) / 2
    fig, ax = plt.subplots(figsize=(10, 6))
    for k, scheme in enumerate(ordered):
        ax.bar(
            x + (k - center) * width,
            customer_utility_by_type(scheme, scenario),
            width=width,
            color=_scheme_color(scheme.key, k),
            edgecolor="grey",
            label=scheme.display_name,
        )
    ax.set_xlabel(r"Customer type $\alpha_i$")
    ax.set_ylabel("Utility ($)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:.2g}" for value in types])
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    return fig, ax


def plot_line_comparison_at_hour(
    schemes: dict[str, SchemeResult],
    scenario,
    *,
    hour_index: int,
    field: str,
    ylabel: str,
    save_path: str | Path | None = None,
    dpi: int = DEFAULT_FIGURE_DPI,
):
    """Line comparison over customer type for reductions or incentives."""

    import matplotlib.pyplot as plt

    if field not in {"D", "lambdas"}:
        raise ValueError("field must be either 'D' or 'lambdas'.")

    _configure_style()
    hour_index = int(np.clip(hour_index, 0, scenario.T - 1))
    fig, ax = plt.subplots(figsize=(12, 6))
    for k, scheme in enumerate(_ordered_schemes(schemes)):
        values = getattr(scheme, field)[:, hour_index]
        ax.plot(
            scenario.alpha,
            values,
            _scheme_marker(scheme.key, k),
            color=_scheme_color(scheme.key, k),
            linewidth=2.5,
            label=scheme.display_name,
        )
    ax.set_xlabel(r"Customer type $\alpha_i$")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    return fig, ax


def plot_load_reduction_stack(
    scenario,
    D: np.ndarray,
    *,
    selected_type: int,
    save_path: str | Path | None = None,
    dpi: int = DEFAULT_FIGURE_DPI,
):
    """Stacked load-composition chart matching the old load/DR figure."""

    import matplotlib.pyplot as plt

    _configure_style()
    selected_type = int(np.clip(selected_type, 0, scenario.N - 1))
    t = np.arange(1, scenario.T + 1)
    reduction = np.asarray(D, dtype=float)[selected_type]
    remaining = scenario.d_n[selected_type] - reduction

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(t, scenario.d_cr[selected_type], color="lightsteelblue", label="critic load")
    ax.bar(
        t,
        scenario.d_sh[selected_type],
        bottom=scenario.d_cr[selected_type],
        color="orange",
        label="shiftable load",
    )
    ax.bar(
        t,
        scenario.d_cu[selected_type],
        bottom=scenario.d_cr[selected_type] + scenario.d_sh[selected_type],
        color="mediumseagreen",
        label="curtailable load",
    )
    ax.bar(t, remaining, width=0.12, color="black", edgecolor="black", hatch="\\\\", label="remaining demand")
    ax.bar(
        t,
        reduction,
        bottom=remaining,
        width=0.12,
        color="gold",
        edgecolor="black",
        hatch="\\\\",
        label="demand reduction",
    )
    ax.plot(t, remaining, color="black", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy (kWh)")
    ax.set_xticks(t)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    return fig, ax


def plot_penalty_execution(
    scheme: SchemeResult,
    scenario,
    *,
    hour_index: int,
    type_indices: list[int] | None = None,
    save_path: str | Path | None = None,
    dpi: int = DEFAULT_FIGURE_DPI,
):
    """Plot utility versus execution level with and without shortfall penalty."""

    import matplotlib.pyplot as plt

    _configure_style()
    hour_index = int(np.clip(hour_index, 0, scenario.T - 1))
    if type_indices is None:
        type_indices = np.unique(np.linspace(0, scenario.N - 1, min(4, scenario.N), dtype=int)).tolist()
    execution = np.linspace(0.0, 1.0, 11)
    colors = ["blue", "purple", "gray", "lightskyblue", "teal"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for k, i in enumerate(type_indices):
        prescribed = float(scheme.D[i, hour_index])
        reward = float(scheme.lambdas[i, hour_index])
        alpha = float(scenario.alpha[i])
        beta = float(scenario.beta[i])
        shortfall_penalty = max(0.0, 2.0 * alpha * prescribed + beta - reward)

        delivered = execution * prescribed
        utility_without_penalty = reward * delivered - alpha * delivered**2 - beta * delivered
        utility_with_penalty = utility_without_penalty - (1.0 - execution) * prescribed * shortfall_penalty

        label = rf"$\alpha_i$={scenario.alpha[i]:.2g}"
        color = colors[k % len(colors)]
        ax.plot(execution, utility_with_penalty, linewidth=3.0, color=color, label=f"with penalty {label}")
        ax.plot(execution, utility_without_penalty, "--", linewidth=2.0, color=color, label=f"without penalty {label}")

    ax.set_xlabel(r"Execution level $e$")
    ax.set_ylabel("Utility")
    ax.grid(True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    return fig, ax


def save_robust_comparison_figures(
    scenario,
    f: np.ndarray,
    *,
    output_dir: str | Path,
    baseline_schemes: dict[str, SchemeResult] | None = None,
    robust_radii: tuple[float, ...] = (0.1, 0.8),
    hour_index: int | None = None,
    selected_type: int = 0,
    dpi: int = DEFAULT_FIGURE_DPI,
    verbose: int = 0,
    robust_verbose: bool = False,
) -> dict[str, Path]:
    """Save Parvizi-style comparison figures that include robust contracts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if baseline_schemes is None:
        baseline_schemes = build_baseline_schemes(scenario, f, verbose=verbose)

    robust_schemes = build_robust_schemes(
        scenario,
        f,
        robust_radii=robust_radii,
        verbose=robust_verbose,
    )
    schemes = dict(baseline_schemes)
    schemes.update(robust_schemes)

    if hour_index is None:
        hour_index = min(11, scenario.T - 1)

    paths: dict[str, Path] = {}
    paths["CUs_utility_with_robust"] = output_dir / "CUs_utility_with_robust.pdf"
    plot_customer_utilities(schemes, scenario, save_path=paths["CUs_utility_with_robust"], dpi=dpi)

    paths["demand_reductions_comparison_with_robust"] = output_dir / "demand_reductions_comparison_with_robust.pdf"
    plot_line_comparison_at_hour(
        schemes,
        scenario,
        hour_index=hour_index,
        field="D",
        ylabel="Demand reduction (kWh)",
        save_path=paths["demand_reductions_comparison_with_robust"],
        dpi=dpi,
    )

    paths["incentive_rate_comparison_with_robust"] = output_dir / "incentive_rate_comparison_with_robust.pdf"
    plot_line_comparison_at_hour(
        schemes,
        scenario,
        hour_index=hour_index,
        field="lambdas",
        ylabel="Incentive reward ($)",
        save_path=paths["incentive_rate_comparison_with_robust"],
        dpi=dpi,
    )

    for radius in robust_radii:
        key = robust_scheme_key(float(radius))
        scheme = robust_schemes[key]
        check = compute_contract_utilities(scheme.D, scenario.alpha, scenario.beta)

        paths[f"{key}_ic"] = output_dir / f"{key}_ic.pdf"
        plot_ic_curves(
            check,
            title=f"Robust Proposed Contract: IC Check, r={float(radius):g}",
            save_path=paths[f"{key}_ic"],
            dpi=dpi,
        )

        paths[f"{key}_load_reduction"] = output_dir / f"{key}_load_reduction_type_{selected_type + 1}.pdf"
        plot_load_reduction_stack(
            scenario,
            scheme.D,
            selected_type=selected_type,
            save_path=paths[f"{key}_load_reduction"],
            dpi=dpi,
        )

    _close_all_figures()
    return paths


def _empty_metric_dict() -> dict[str, list[float]]:
    return {key: [] for key in SCHEME_ORDER}


def run_demand_scale_experiment(
    scenario,
    f: np.ndarray,
    *,
    scales: np.ndarray,
    verbose: int = 0,
) -> ScaleExperimentResult:
    """Solve the five schemes while scaling the hourly demand requirement."""

    scales = np.asarray(scales, dtype=float)
    operator_values = _empty_metric_dict()
    customer_values = _empty_metric_dict()
    welfare_values = _empty_metric_dict()

    for scale in scales:
        scaled_scenario = replace(scenario, Dreq=scenario.Dreq * float(scale))
        schemes = build_baseline_schemes(scaled_scenario, f, verbose=verbose)
        for key in SCHEME_ORDER:
            scheme = schemes[key]
            go = operator_utility(scheme, scaled_scenario, f)
            cu = total_customer_utility(scheme, scaled_scenario, f)
            operator_values[key].append(go)
            customer_values[key].append(cu)
            welfare_values[key].append(go + cu)

    return ScaleExperimentResult(
        x_values=scales,
        operator_utility={key: np.asarray(values) for key, values in operator_values.items()},
        customer_utility={key: np.asarray(values) for key, values in customer_values.items()},
        social_welfare={key: np.asarray(values) for key, values in welfare_values.items()},
    )


def run_customer_scale_experiment(
    scenario,
    f: np.ndarray,
    *,
    customer_counts: np.ndarray,
    verbose: int = 0,
) -> ScaleExperimentResult:
    """Solve the five schemes while varying the number of customers ``M``."""

    customer_counts = np.asarray(customer_counts, dtype=int)
    operator_values = _empty_metric_dict()
    customer_values = _empty_metric_dict()
    welfare_values = _empty_metric_dict()

    for M in customer_counts:
        scaled_scenario = replace(scenario, M=int(M))
        schemes = build_baseline_schemes(scaled_scenario, f, verbose=verbose)
        for key in SCHEME_ORDER:
            scheme = schemes[key]
            go = operator_utility(scheme, scaled_scenario, f)
            cu = total_customer_utility(scheme, scaled_scenario, f)
            operator_values[key].append(go)
            customer_values[key].append(cu)
            welfare_values[key].append(go + cu)

    return ScaleExperimentResult(
        x_values=customer_counts.astype(float),
        operator_utility={key: np.asarray(values) for key, values in operator_values.items()},
        customer_utility={key: np.asarray(values) for key, values in customer_values.items()},
        social_welfare={key: np.asarray(values) for key, values in welfare_values.items()},
    )


def plot_scale_bars(
    x_values: np.ndarray,
    values_by_scheme: dict[str, np.ndarray],
    *,
    xlabel: str,
    ylabel: str,
    save_path: str | Path | None = None,
    dpi: int = DEFAULT_FIGURE_DPI,
):
    """Grouped bar chart for demand/customer scaling experiments."""

    import matplotlib.pyplot as plt

    _configure_style()
    x_values = np.asarray(x_values)
    x = np.arange(x_values.size)
    width = 0.15
    fig, ax = plt.subplots(figsize=(8, 8))
    for k, key in enumerate(SCHEME_ORDER):
        ax.bar(
            x + k * width,
            np.asarray(values_by_scheme[key], dtype=float),
            width=width,
            color=SCHEME_COLORS[key],
            edgecolor="grey",
            label=SCHEME_LABELS[key],
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x + width * (len(SCHEME_ORDER) - 1) / 2)
    ax.set_xticklabels([f"{value:g}" for value in x_values])
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    return fig, ax


def run_demand_prediction_sensitivity(
    scenario,
    f: np.ndarray,
    *,
    percent_changes: np.ndarray,
    verbose: int = 0,
) -> SensitivityResult:
    """Evaluate proposed-contract sensitivity to Dreq prediction error."""

    percent_changes = np.asarray(percent_changes, dtype=float)
    baseline_schemes = build_baseline_schemes(scenario, f, verbose=verbose)
    baseline = baseline_schemes["proposed"]
    baseline_operator = operator_utility(baseline, scenario, f)
    baseline_reduction = np.sum(baseline.D, axis=1)

    operator_errors = []
    reduction_errors = []
    for change in percent_changes:
        estimated = replace(scenario, Dreq=scenario.Dreq * (1.0 + change / 100.0))
        scheme = build_baseline_schemes(estimated, f, verbose=verbose)["proposed"]
        operator_errors.append(100.0 * (operator_utility(scheme, estimated, f) - baseline_operator) / abs(baseline_operator))
        reduction_errors.append(100.0 * (np.sum(scheme.D, axis=1) - baseline_reduction) / np.maximum(np.abs(baseline_reduction), 1e-12))

    return SensitivityResult(
        percent_changes=percent_changes,
        operator_error_percent=np.asarray(operator_errors, dtype=float),
        demand_reduction_error_percent=np.asarray(reduction_errors, dtype=float).T,
    )


def plot_operator_error(
    sensitivity: SensitivityResult,
    *,
    threshold: float | None = 0.0,
    save_path: str | Path | None = None,
    dpi: int = DEFAULT_FIGURE_DPI,
):
    """Plot percent error in grid/operator utility."""

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    _configure_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    if threshold is not None:
        ax.axvspan(sensitivity.percent_changes[0], threshold, facecolor="red", alpha=0.18)
        ax.axvspan(threshold, sensitivity.percent_changes[-1], facecolor="green", alpha=0.14)
    ax.plot(sensitivity.percent_changes, sensitivity.operator_error_percent, linewidth=3.0)
    ax.set_xlabel("Percent error in demand prediction")
    ax.set_ylabel("Percent error in GO utility")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xticks(sensitivity.percent_changes)
    ax.grid(True)
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    return fig, ax


def plot_reduction_error(
    sensitivity: SensitivityResult,
    scenario,
    *,
    threshold: float | None = 0.0,
    save_path: str | Path | None = None,
    dpi: int = DEFAULT_FIGURE_DPI,
):
    """Plot percent error in each customer type's total demand reduction."""

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    _configure_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    if threshold is not None:
        ax.axvspan(sensitivity.percent_changes[0], threshold, facecolor="red", alpha=0.18)
        ax.axvspan(threshold, sensitivity.percent_changes[-1], facecolor="green", alpha=0.14)
    for i in range(sensitivity.demand_reduction_error_percent.shape[0]):
        ax.plot(
            sensitivity.percent_changes,
            sensitivity.demand_reduction_error_percent[i],
            linewidth=2.5,
            label=rf"$\alpha_i$={scenario.alpha[i]:.2g}",
        )
    ax.set_xlabel("Percent error in demand prediction")
    ax.set_ylabel("Percent error in D")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xticks(sensitivity.percent_changes)
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    return fig, ax


def run_type_timing_experiment(
    *,
    type_counts: np.ndarray,
    M: int = 80,
    T: int = 24,
    price_scale: float = 2.0,
    dreq_low: float = 100.0,
    dreq_high: float = 200.0,
    seed: int = 42,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure proposed-contract solve time for available type counts."""

    type_counts = np.asarray(type_counts, dtype=int)
    times = []
    for N in type_counts:
        alpha = np.linspace(1.0, 1.4, N)
        beta = np.linspace(0.5, 0.7, N)
        f = np.ones(N, dtype=float) / N
        scenario = build_load_scenario(
            M=M,
            T=T,
            N=int(N),
            alpha=alpha,
            beta=beta,
            price_scale=price_scale,
            dreq_low=dreq_low,
            dreq_high=dreq_high,
            seed=seed,
        )
        start = perf_counter()
        solve_proposed_contract(
            M=scenario.M,
            T=scenario.T,
            N=scenario.N,
            Krt=scenario.Krt,
            Ks=scenario.Ks,
            Dreq=scenario.Dreq,
            pi=scenario.pi,
            alpha=scenario.alpha,
            beta=scenario.beta,
            f=f,
            verbose=verbose,
        )
        times.append(perf_counter() - start)
    return type_counts, np.asarray(times, dtype=float)


def plot_type_timing(
    type_counts: np.ndarray,
    times: np.ndarray,
    *,
    save_path: str | Path | None = None,
    dpi: int = DEFAULT_FIGURE_DPI,
):
    """Plot computational time versus number of types."""

    import matplotlib.pyplot as plt

    _configure_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(type_counts, times, linewidth=3.0, marker="o")
    for n_types, elapsed in zip(type_counts, times):
        ax.annotate(f"{elapsed:.2f} s", (n_types, elapsed), textcoords="offset points", xytext=(0, 10), ha="center")
    ax.set_xlabel("Number of Types")
    ax.set_ylabel("Computational Time (s)")
    ax.set_xticks(type_counts)
    ax.grid(True)
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    return fig, ax


def save_all_parvizi_figures(
    scenario,
    f: np.ndarray,
    *,
    output_dir: str | Path,
    dpi: int = DEFAULT_FIGURE_DPI,
    verbose: int = 0,
    include_robust: bool = True,
    robust_radii: tuple[float, ...] = (0.1, 0.8),
) -> dict[str, Path]:
    """Generate the adapted Parvizi PDFs and optional robust companion figures."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    schemes = build_baseline_schemes(scenario, f, verbose=verbose)
    hour_index = min(11, scenario.T - 1)
    selected_type = min(6, scenario.N - 1)

    paths["CUs_utility"] = output_dir / "CUs_utility.pdf"
    plot_customer_utilities(schemes, scenario, save_path=paths["CUs_utility"], dpi=dpi)

    paths["demand_reductions_comparison"] = output_dir / "demand_reductions_comparison.pdf"
    plot_line_comparison_at_hour(
        schemes,
        scenario,
        hour_index=hour_index,
        field="D",
        ylabel="Demand reduction (kWh)",
        save_path=paths["demand_reductions_comparison"],
        dpi=dpi,
    )

    paths["incentive_rate_comparison"] = output_dir / "incentive_rate_comparison.pdf"
    plot_line_comparison_at_hour(
        schemes,
        scenario,
        hour_index=hour_index,
        field="lambdas",
        ylabel="Incentive reward ($)",
        save_path=paths["incentive_rate_comparison"],
        dpi=dpi,
    )

    paths["loads demand reduction Maryam n=7"] = output_dir / "loads demand reduction Maryam n=7.pdf"
    plot_load_reduction_stack(scenario, schemes["proposed"].D, selected_type=selected_type, save_path=paths["loads demand reduction Maryam n=7"], dpi=dpi)

    paths["Penalty"] = output_dir / "Penalty.pdf"
    plot_penalty_execution(schemes["proposed"], scenario, hour_index=hour_index, save_path=paths["Penalty"], dpi=dpi)

    if include_robust:
        robust_paths = save_robust_comparison_figures(
            scenario,
            f,
            output_dir=output_dir / "Robust",
            baseline_schemes=schemes,
            robust_radii=robust_radii,
            hour_index=hour_index,
            selected_type=0,
            dpi=dpi,
            verbose=verbose,
        )
        paths.update({f"Robust {name}": path for name, path in robust_paths.items()})

    demand_dir = output_dir / "ScaleInDemandReduction"
    demand_scales = np.array([0.25, 0.50, 1.00, 1.10, 1.25, 1.50, 2.00])
    demand_result = run_demand_scale_experiment(scenario, f, scales=demand_scales, verbose=verbose)
    paths["GOU_vs_Dreq"] = demand_dir / "GOU_vs_Dreq.pdf"
    plot_scale_bars(demand_result.x_values, demand_result.operator_utility, xlabel="Scaled Demand Require", ylabel="GO Utility ($)", save_path=paths["GOU_vs_Dreq"], dpi=dpi)
    paths["CUU_vs_Dreq"] = demand_dir / "CUU_vs_Dreq.pdf"
    plot_scale_bars(demand_result.x_values, demand_result.customer_utility, xlabel="Scaled Demand Require", ylabel="CUs Utility ($)", save_path=paths["CUU_vs_Dreq"], dpi=dpi)
    paths["SocialWelfare_vs_Dreq"] = demand_dir / "SocialWelfare_vs_Dreq.pdf"
    plot_scale_bars(demand_result.x_values, demand_result.social_welfare, xlabel="Scaled Demand Require", ylabel="Social Welfare ($)", save_path=paths["SocialWelfare_vs_Dreq"], dpi=dpi)

    customer_dir = output_dir / "ScaleInNumberOfCustomers"
    customer_counts = np.array([10, 20, 50, 100, 200, 500])
    customer_result = run_customer_scale_experiment(scenario, f, customer_counts=customer_counts, verbose=verbose)
    paths["GOU_vs_M"] = customer_dir / "GOU_vs_M.pdf"
    plot_scale_bars(customer_result.x_values, customer_result.operator_utility, xlabel="Number of Customers M", ylabel="GO Utility ($)", save_path=paths["GOU_vs_M"], dpi=dpi)
    paths["CUU_vs_M"] = customer_dir / "CUU_vs_M.pdf"
    plot_scale_bars(customer_result.x_values, customer_result.customer_utility, xlabel="Number of Customers M", ylabel="CUs Utility ($)", save_path=paths["CUU_vs_M"], dpi=dpi)
    paths["SocialWelfare_vs_M"] = customer_dir / "SocialWelfare_vs_M.pdf"
    plot_scale_bars(customer_result.x_values, customer_result.social_welfare, xlabel="Number of Customers M", ylabel="Social Welfare ($)", save_path=paths["SocialWelfare_vs_M"], dpi=dpi)

    sensitivity_dir = output_dir / "Sensitivity analysis"
    broad_changes = np.array([-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10], dtype=float)
    broad_sensitivity = run_demand_prediction_sensitivity(scenario, f, percent_changes=broad_changes, verbose=verbose)
    paths["Error_vs_GOU"] = sensitivity_dir / "Error_vs_GOU.pdf"
    plot_operator_error(broad_sensitivity, threshold=0.0, save_path=paths["Error_vs_GOU"], dpi=dpi)
    paths["Error_vs_DemandReduction"] = sensitivity_dir / "Error_vs_DemandReduction.pdf"
    plot_reduction_error(broad_sensitivity, scenario, threshold=0.0, save_path=paths["Error_vs_DemandReduction"], dpi=dpi)

    fine_dir = output_dir / "Untitled Folder"
    fine_changes = np.arange(-10, 11, 1, dtype=float)
    fine_sensitivity = run_demand_prediction_sensitivity(scenario, f, percent_changes=fine_changes, verbose=verbose)
    paths["Untitled Folder Error_vs_GOU"] = fine_dir / "Error_vs_GOU.pdf"
    plot_operator_error(fine_sensitivity, threshold=0.0, save_path=paths["Untitled Folder Error_vs_GOU"], dpi=dpi)

    timing_dir = output_dir / "new"
    type_counts = np.array([3, 4, 5, 6])
    type_counts, times = run_type_timing_experiment(type_counts=type_counts, M=scenario.M, T=scenario.T, verbose=verbose)
    paths["Times_vs_Types"] = timing_dir / "Times_vs_Types.pdf"
    plot_type_timing(type_counts, times, save_path=paths["Times_vs_Types"], dpi=dpi)

    _close_all_figures()
    return paths
