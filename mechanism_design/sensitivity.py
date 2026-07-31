"""Sensitivity analysis for misspecified physical contract constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract_models import ContractSolution, solve_proposed_contract


@dataclass
class ConstraintSensitivityPoint:
    """One point in a `Ks` or `Krt` misspecification sweep."""

    parameter: str
    factor: float
    solver_success: bool
    feasible_under_true_caps: bool
    objective: float
    weighted_average_reduction: float
    max_hourly_cap_violation: float
    max_daily_cap_violation: float
    message: str = ""

    @property
    def feasible(self) -> bool:
        """True only when the estimated problem solved and the result is physically feasible."""

        return self.solver_success and self.feasible_under_true_caps


def _physical_cap_violations(D: np.ndarray, true_Krt: np.ndarray, true_Ks: float) -> tuple[float, float]:
    """Return positive violations of the true hourly and daily caps."""

    hourly_violation = float(np.max(D - true_Krt.reshape(1, -1)))
    daily_violation = float(np.max(np.sum(D, axis=1) - true_Ks))
    return max(0.0, hourly_violation), max(0.0, daily_violation)


def _weighted_average_reduction(D: np.ndarray, f: np.ndarray) -> float:
    """Distribution-weighted average hourly customer reduction."""

    per_type_average = np.sum(D, axis=1) / D.shape[1]
    return float(np.dot(f, per_type_average))


def evaluate_constraint_misestimation(
    *,
    scenario,
    f: np.ndarray,
    parameter: str,
    factor: float,
    verbose: int = 0,
) -> ConstraintSensitivityPoint:
    """
    Solve the proposed contract with one misspecified cap and check true feasibility.

    `parameter="Ks"` means the daily cap used by the optimization is
    `factor * scenario.Ks`. `parameter="Krt"` means every hourly cap is
    multiplied by `factor`. Feasibility is then checked against the original
    physical caps from `scenario`.
    """

    parameter = parameter.strip()
    if parameter not in {"Ks", "Krt"}:
        raise ValueError("parameter must be either 'Ks' or 'Krt'.")

    estimated_Ks = scenario.Ks * factor if parameter == "Ks" else scenario.Ks
    estimated_Krt = scenario.Krt * factor if parameter == "Krt" else scenario.Krt

    try:
        solution: ContractSolution = solve_proposed_contract(
            M=scenario.M,
            T=scenario.T,
            N=scenario.N,
            Krt=estimated_Krt,
            Ks=estimated_Ks,
            Dreq=scenario.Dreq,
            pi=scenario.pi,
            alpha=scenario.alpha,
            beta=scenario.beta,
            f=f,
            verbose=verbose,
        )
    except Exception as exc:
        return ConstraintSensitivityPoint(
            parameter=parameter,
            factor=float(factor),
            solver_success=False,
            feasible_under_true_caps=False,
            objective=np.nan,
            weighted_average_reduction=np.nan,
            max_hourly_cap_violation=np.nan,
            max_daily_cap_violation=np.nan,
            message=str(exc),
        )

    hourly_violation, daily_violation = _physical_cap_violations(solution.D, scenario.Krt, scenario.Ks)
    feasible_under_true_caps = hourly_violation <= 1e-7 and daily_violation <= 1e-7

    return ConstraintSensitivityPoint(
        parameter=parameter,
        factor=float(factor),
        solver_success=True,
        feasible_under_true_caps=feasible_under_true_caps,
        objective=float(solution.objective),
        weighted_average_reduction=_weighted_average_reduction(solution.D, f),
        max_hourly_cap_violation=hourly_violation,
        max_daily_cap_violation=daily_violation,
        message=solution.status,
    )


def run_constraint_misestimation_sensitivity(
    *,
    scenario,
    f: np.ndarray,
    factors: np.ndarray,
    parameters: tuple[str, ...] = ("Ks", "Krt"),
    verbose: int = 0,
) -> list[ConstraintSensitivityPoint]:
    """Run a misspecification sweep for `Ks` and/or `Krt`."""

    f = np.asarray(f, dtype=float)
    f = f / np.sum(f)
    factors = np.asarray(factors, dtype=float)

    points: list[ConstraintSensitivityPoint] = []
    for parameter in parameters:
        for factor in factors:
            points.append(
                evaluate_constraint_misestimation(
                    scenario=scenario,
                    f=f,
                    parameter=parameter,
                    factor=float(factor),
                    verbose=verbose,
                )
            )
    return points
