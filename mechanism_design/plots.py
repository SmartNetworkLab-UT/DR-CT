"""Plotting helpers for notebook outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_FIGURE_DPI = 600


def save_figure(fig, path: str | Path, *, dpi: int = DEFAULT_FIGURE_DPI) -> Path:
    """Save a matplotlib figure as a high-resolution PDF."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=dpi, bbox_inches="tight")
    return output_path


def plot_ic_curves(utility_check, *, title: str = "IC Check", save_path: str | Path | None = None, dpi: int = DEFAULT_FIGURE_DPI):
    """Plot utility for each true type as a function of the reported contract."""

    import matplotlib.pyplot as plt

    n_types = utility_check.utility_total.shape[0]
    types = np.arange(1, n_types + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_types):
        ax.plot(types, utility_check.utility_total[i], marker="o", label=f"True Type i={i + 1}")
    ax.scatter(types, utility_check.truthful_total, s=100, zorder=3, label="Truthful Utility")
    ax.set_xlabel("Reported Contract Type (j)")
    ax.set_ylabel("Total Utility of True Type i")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    return fig, ax


def plot_load_reduction(
    scenario,
    D: np.ndarray,
    *,
    selected_type: int = 0,
    title_prefix: str = "Type",
    save_path: str | Path | None = None,
    dpi: int = DEFAULT_FIGURE_DPI,
):
    """Plot load components, applied DR, and remaining demand for one type."""

    import matplotlib.pyplot as plt

    t = np.arange(1, scenario.T + 1)
    baseline = scenario.d_n[selected_type]
    crit = scenario.d_cr[selected_type]
    curt = scenario.d_cu[selected_type]
    shift = scenario.d_sh[selected_type]
    reduction = np.asarray(D)[selected_type]
    remaining = baseline - reduction

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(t, crit, label="critical load")
    ax.bar(t, curt, bottom=crit, label="curtailable load")
    ax.bar(t, shift, bottom=crit + curt, label="shiftable load")
    ax.bar(
        t,
        reduction,
        bottom=crit + curt + shift,
        fill=False,
        hatch="///",
        linewidth=1.5,
        label="demand reduction",
    )
    ax.plot(t, remaining, marker="o", linewidth=2, label="remaining demand")
    ax.set_xlabel("Time (hour)")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title(
        f"{title_prefix} (alpha,beta)=({scenario.alpha[selected_type]:.1f},"
        f"{scenario.beta[selected_type]:.2f}): load composition and DR"
    )
    ax.set_xticks(t)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    return fig, ax


def plot_type_bars(
    values_by_scenario: dict[str, np.ndarray],
    *,
    ylabel: str,
    title: str,
    save_path: str | Path | None = None,
    dpi: int = DEFAULT_FIGURE_DPI,
):
    """Grouped bar chart over contract types."""

    import matplotlib.pyplot as plt

    labels = list(values_by_scenario)
    values = [np.asarray(values_by_scenario[label], dtype=float).reshape(-1) for label in labels]
    n_types = values[0].size
    x = np.arange(n_types)
    width = min(0.8 / max(len(values), 1), 0.18)

    fig, ax = plt.subplots(figsize=(10, 6))
    center = (len(values) - 1) / 2
    for k, (label, series) in enumerate(zip(labels, values)):
        ax.bar(x + (k - center) * width, series, width, label=label)

    ax.set_xlabel("CU type")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(np.arange(1, n_types + 1))
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    return fig, ax


def plot_type_lines(
    values_by_scenario: dict[str, np.ndarray],
    *,
    ylabel: str,
    title: str,
    save_path: str | Path | None = None,
    dpi: int = DEFAULT_FIGURE_DPI,
):
    """Line chart over contract types."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, values in values_by_scenario.items():
        values = np.asarray(values, dtype=float).reshape(-1)
        types = np.arange(1, values.size + 1)
        ax.plot(types, values, marker="o", label=label)

    ax.set_xlabel("CU type")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    return fig, ax


