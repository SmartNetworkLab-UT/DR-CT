"""Data loading and scenario construction for the DR contract notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATASET_DIR_NAME = "Dataset on Hourly Load Profiles for 8 Facilities (8760 hours)"


@dataclass
class LoadScenario:
    """All exogenous inputs shared by the contract optimization models."""

    M: int
    T: int
    N: int
    folder: Path
    files: list[Path]
    alpha: np.ndarray
    beta: np.ndarray
    d_n: np.ndarray
    d_cr: np.ndarray
    d_cu: np.ndarray
    d_sh: np.ndarray
    Krt: np.ndarray
    Ks: float
    pi: np.ndarray
    Dreq: np.ndarray
    Dreq_upper_bound: np.ndarray

    @property
    def feasibility_report(self) -> dict[str, float | bool]:
        sum_krt = float(np.sum(self.Krt))
        return {
            "sum_Krt": sum_krt,
            "Ks": float(self.Ks),
            "gap": float(self.Ks - sum_krt),
            "daily_cap_covers_hourly_caps": bool(self.Ks >= sum_krt),
        }


def default_dataset_dir(base_dir: str | Path | None = None) -> Path:
    """Return the dataset directory that is checked into this repository."""

    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1]
    return Path(base_dir) / DATASET_DIR_NAME


def load_hourly_profiles(
    folder: str | Path,
    n_types: int,
    hours: int = 24,
    column: int = 0,
) -> tuple[np.ndarray, list[Path]]:
    """
    Load the first `hours` numeric rows from column A of the first `n_types` CSVs.

    The original notebooks used rows A2:A25. With the checked-in CSV files, that
    is equivalent to skipping the header row and reading the next 24 values.
    """

    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))
    if len(files) < n_types:
        raise ValueError(f"Expected at least {n_types} CSV files in {folder}, found {len(files)}.")

    selected_files = files[:n_types]
    profiles: list[np.ndarray] = []
    for path in selected_files:
        values = np.genfromtxt(
            path,
            delimiter=",",
            skip_header=1,
            max_rows=hours,
            usecols=column,
            dtype=float,
        )
        values = np.atleast_1d(values).astype(float)
        if values.size != hours or np.any(np.isnan(values)):
            raise ValueError(f"{path} does not contain {hours} numeric load values in column {column}.")
        profiles.append(values)

    return np.vstack(profiles), selected_files


def decompose_loads(
    d_n: np.ndarray,
    critical_share: float = 0.60,
    curtailable_share: float = 0.37,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each load profile into critical, curtailable, and shiftable parts."""

    d_n = np.asarray(d_n, dtype=float)
    d_cr = critical_share * d_n
    d_cu = curtailable_share * d_n
    d_sh = d_n - d_cr - d_cu
    return d_cr, d_cu, d_sh


def compute_capacity_limits(d_n: np.ndarray, d_cr: np.ndarray, d_cu: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute the hourly and daily DR caps used by the notebooks.

    `Krt[t]` is the conservative hourly cap across types, and `Ks` is the
    conservative daily cap from curtailable energy.
    """

    Krt = np.min(d_n - d_cr, axis=0)
    Ks = float(np.min(np.sum(d_cu, axis=1)))
    return Krt, Ks


def build_tou_prices(
    hours: int,
    scale: float = 2.0,
    base: float = 3.5,
    amplitude: float = 0.25,
    peak_hour: int = 13,
) -> np.ndarray:
    """Build the sinusoidal time-of-use price vector used in the experiments."""

    hour_index = np.arange(1, hours + 1)
    return scale * (base + amplitude * np.sin((hour_index - peak_hour) / hours * 2 * np.pi))


def build_dreq_random(
    Krt: np.ndarray,
    M: int,
    N: int,
    low: float,
    high: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a feasible hourly demand-reduction requirement.

    Each requirement is sampled from `[low, min(high, M*N*Krt[t])]`.
    """

    Krt = np.asarray(Krt, dtype=float)
    upper = np.minimum(high, M * N * Krt)
    bad = np.where(upper < low)[0]
    if bad.size:
        raise ValueError(f"Infeasible hours, 1-based indexing: {bad + 1}. Upper bound is below {low}.")

    rng = np.random.default_rng(seed)
    Dreq = rng.uniform(low, upper)
    validate_dreq(Dreq, Krt, M, N, low=low, high=high)
    return Dreq, upper


def validate_dreq(
    Dreq: np.ndarray,
    Krt: np.ndarray,
    M: int,
    N: int,
    low: float | None = None,
    high: float | None = None,
) -> None:
    """Raise if `Dreq` is outside the feasible hourly range."""

    Dreq = np.asarray(Dreq, dtype=float)
    Krt = np.asarray(Krt, dtype=float)
    if np.any(Dreq < -1e-12) or np.any(Dreq > M * N * Krt + 1e-9):
        raise ValueError("Dreq must satisfy 0 <= Dreq[t] <= M*N*Krt[t] for every hour.")
    if low is not None and np.any(Dreq < low - 1e-9):
        raise ValueError(f"Dreq contains values below the requested lower bound {low}.")
    if high is not None and np.any(Dreq > high + 1e-9):
        raise ValueError(f"Dreq contains values above the requested upper bound {high}.")


def build_load_scenario(
    *,
    M: int = 80,
    T: int = 24,
    N: int = 5,
    folder: str | Path | None = None,
    alpha: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    critical_share: float = 0.60,
    curtailable_share: float = 0.37,
    price_scale: float = 2.0,
    dreq_low: float = 100.0,
    dreq_high: float = 200.0,
    seed: int = 42,
) -> LoadScenario:
    """Build the full input bundle used by the notebooks."""

    folder_path = default_dataset_dir() if folder is None else Path(folder)
    if alpha is None:
        alpha = np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=float)
    if beta is None:
        beta = np.array([0.5, 0.55, 0.6, 0.65, 0.7], dtype=float)

    alpha = np.asarray(alpha, dtype=float).reshape(N)
    beta = np.asarray(beta, dtype=float).reshape(N)
    d_n, files = load_hourly_profiles(folder_path, n_types=N, hours=T)
    d_cr, d_cu, d_sh = decompose_loads(d_n, critical_share, curtailable_share)
    Krt, Ks = compute_capacity_limits(d_n, d_cr, d_cu)
    pi = build_tou_prices(T, scale=price_scale)
    Dreq, upper = build_dreq_random(Krt, M=M, N=N, low=dreq_low, high=dreq_high, seed=seed)

    return LoadScenario(
        M=M,
        T=T,
        N=N,
        folder=folder_path,
        files=files,
        alpha=alpha,
        beta=beta,
        d_n=d_n,
        d_cr=d_cr,
        d_cu=d_cu,
        d_sh=d_sh,
        Krt=Krt,
        Ks=Ks,
        pi=pi,
        Dreq=Dreq,
        Dreq_upper_bound=upper,
    )
