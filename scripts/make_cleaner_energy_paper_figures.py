"""Create the cleaner-energy evidence figures from reproducible assessment outputs.

The figures distinguish final 30-day assessments from the exploratory capacity
screen. They report electrical-simulation metrics only; the normalised asset
index is not a market cost or a life-cycle footprint measure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPOSITORY_ROOT / "data" / "results" / "cleaner_energy"
LOW = RESULTS / "resource_efficient" / "best_plans.csv"
HIGH = RESULTS / "high_redundancy" / "best_plans.csv"
ENVELOPE = RESULTS / "pv_envelope" / "best_plans.csv"
CONTROLLERS = RESULTS / "controller_comparison" / "best_plans.csv"
OUT = REPOSITORY_ROOT / "figures" / "generated"

BLUE = "#176B87"
ORANGE = "#D68635"
GREEN = "#4C956C"
RED = "#C8543C"
GREY = "#6D6D6D"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.linewidth": 0.9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def _film_row(path: Path, pv_scale: float = 1.0) -> pd.Series:
    data = pd.read_csv(path)
    rows = data[(data["controller"] == "film") & np.isclose(data["pv_generation_scale"], pv_scale)]
    if len(rows) != 1:
        raise ValueError(f"Expected one FiLM row for PV scale {pv_scale} in {path}")
    return rows.iloc[0]


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".png": {"dpi": 360},
        ".pdf": {},
        ".svg": {},
    }.items():
        fig.savefig(OUT / f"{stem}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def resource_efficiency_comparison() -> None:
    """Show the primary evidence: same evaluated security, less resource use."""
    low = _film_row(LOW)
    high = _film_row(HIGH)
    names = ["Resource-efficient\nconfiguration", "High-redundancy\nconfiguration"]
    colours = [BLUE, RED]
    cost = [float(low["asset_cost_index"]), float(high["asset_cost_index"])]
    loss = [float(low["annual_loss_mwh"]), float(high["annual_loss_mwh"])]
    cost_reduction = (1.0 - cost[0] / cost[1]) * 100.0
    loss_reduction = (1.0 - loss[0] / loss[1]) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), constrained_layout=True)
    for panel, ax, values, ylabel, reduction in [
        ("a", axes[0], cost, "Normalised asset-cost index", cost_reduction),
        ("b", axes[1], loss, "Annualised line-loss estimate (MWh)", loss_reduction),
    ]:
        bars = ax.bar(names, values, color=colours, width=0.62, edgecolor="white", linewidth=0.8)
        _style_axis(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(f"({panel}) {reduction:.1f}% lower", loc="left", fontweight="bold")
        ymax = max(values) * 1.20
        ax.set_ylim(0, ymax)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + ymax * 0.025, f"{value:.1f}", ha="center", va="bottom", fontweight="bold")
    fig.suptitle("Same PV profile and 30 held-out daily scenarios; both plans had zero observed day-level events", y=1.03, fontsize=9.5)
    _save(fig, "fig_resource_efficiency")


def pv_feasibility_envelope() -> None:
    """Make the tested feasibility boundary explicit without claiming universality."""
    data = pd.read_csv(ENVELOPE)
    labels = {"film": "FiLM-PPO co-planning", "droop": "Local Volt-VAR droop"}
    order = [controller for controller in ("film", "droop") if controller in set(data["controller"])]
    fig, ax = plt.subplots(figsize=(6.8, 2.65), constrained_layout=True)
    for y, controller in enumerate(order):
        subset = data[data["controller"] == controller].sort_values("pv_generation_scale")
        feasible = subset[subset["feasible"] == 1]
        infeasible = subset[subset["feasible"] == 0]
        ax.scatter(feasible["pv_generation_scale"], np.full(len(feasible), y), s=64, color=BLUE, marker="o", zorder=3, label="Feasible plan found" if y == 0 else None)
        ax.scatter(infeasible["pv_generation_scale"], np.full(len(infeasible), y), s=64, color=RED, marker="x", linewidths=1.8, zorder=3, label="No feasible plan found" if y == 0 else None)
    ax.set_yticks(range(len(order)), [labels[item] for item in order])
    ax.set_xlabel("PV generation scale relative to the base profile")
    ax.set_xlim(0.72, 1.28)
    ax.set_xticks([0.8, 1.0, 1.2])
    ax.set_title("Empirical feasibility over the evaluated capacity grid", loc="left", fontweight="bold")
    ax.grid(True, axis="x", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.30), frameon=False)
    _save(fig, "fig_pv_feasibility_envelope")


def fixed_hardware_controller_comparison() -> None:
    """Separate the controller result from the outer planning result."""
    data = pd.read_csv(CONTROLLERS).set_index("controller")
    order = ["film", "blind", "concat"]
    names = ["FiLM-PPO", "Blind PPO", "Concat PPO"]
    colours = [BLUE, GREEN, ORANGE]
    annual_loss = [float(data.loc[item, "annual_loss_mwh"]) for item in order]
    risk = [float(data.loc[item, "pr_violation_any"]) for item in order]
    reduction = (1.0 - annual_loss[0] / annual_loss[1]) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), constrained_layout=True)
    bars = axes[0].bar(names, annual_loss, color=colours, width=0.62, edgecolor="white", linewidth=0.8)
    _style_axis(axes[0])
    axes[0].set_ylabel("Annualised line-loss estimate (MWh)")
    axes[0].set_title(f"(a) FiLM-PPO is {reduction:.1f}% lower than Blind PPO", loc="left", fontweight="bold")
    axes[0].set_ylim(0, max(annual_loss) * 1.20)
    for bar, value in zip(bars, annual_loss):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + max(annual_loss) * 0.025, f"{value:.1f}", ha="center", va="bottom", fontweight="bold")

    axes[1].set_title("(b) Same observed safety outcome", loc="left", fontweight="bold")
    axes[1].set_xlim(-0.5, 2.5)
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].set_xticks(range(3), names)
    axes[1].set_yticks([])
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["left"].set_visible(False)
    axes[1].spines["bottom"].set_visible(False)
    axes[1].text(0.5, 0.85, "Days with at least one voltage event", transform=axes[1].transAxes, ha="center", va="center", color=GREY)
    for x, (name, colour, value) in enumerate(zip(names, colours, risk)):
        face = "#EAF4EC" if np.isclose(value, 0.0) else "#FDEBE7"
        edge = GREEN if np.isclose(value, 0.0) else RED
        axes[1].text(x, 0.0, f"{int(round(value * 30))} / 30", ha="center", va="center", fontsize=15, fontweight="bold", color=edge,
                     bbox={"boxstyle": "round,pad=0.52", "facecolor": face, "edgecolor": edge, "linewidth": 1.2})
    fig.suptitle("Fixed hardware, identical PV profile and 30 held-out daily scenarios", y=1.03, fontsize=9.5)
    _save(fig, "fig_fixed_hardware_controller_comparison")


def write_source_data() -> None:
    """Provide a compact, reviewable source-data table for the three figures."""
    low = _film_row(LOW)
    high = _film_row(HIGH)
    controller = pd.read_csv(CONTROLLERS)
    envelope = pd.read_csv(ENVELOPE)
    rows = []
    for name, row in [("resource_efficient", low), ("high_redundancy", high)]:
        rows.append({"figure": "resource_efficiency", "series": name, "asset_cost_index": row["asset_cost_index"], "annual_loss_mwh": row["annual_loss_mwh"], "observed_day_event_rate": row["pr_violation_any"]})
    for _, row in controller.iterrows():
        rows.append({"figure": "fixed_hardware_controller", "series": row["controller"], "asset_cost_index": row["asset_cost_index"], "annual_loss_mwh": row["annual_loss_mwh"], "observed_day_event_rate": row["pr_violation_any"]})
    for _, row in envelope.iterrows():
        rows.append({"figure": "pv_feasibility_envelope", "series": row["controller"], "pv_generation_scale": row["pv_generation_scale"], "feasible": row["feasible"]})
    pd.DataFrame(rows).to_csv(OUT / "cleaner_energy_figure_source_data.csv", index=False)


def main() -> None:
    for path in (LOW, HIGH, ENVELOPE, CONTROLLERS):
        if not path.exists():
            raise FileNotFoundError(f"Required assessment output is missing: {path}")
    resource_efficiency_comparison()
    pv_feasibility_envelope()
    fixed_hardware_controller_comparison()
    write_source_data()
    print(f"Saved cleaner-energy evidence figures to: {OUT}")


if __name__ == "__main__":
    main()
