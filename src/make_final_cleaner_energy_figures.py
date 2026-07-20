"""Create the final Cleaner Energy Systems result figures and source data.

The script uses only frozen evaluation outputs. It also replays one pre-selected
worst day to visualise how the AC safety projection changes the learned action.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import pandas as pd
import torch

from evaluate_crdc_policy import build_env, load_actor
from physics_safety_projection import ACPowerFlowSafetyProjector


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "generated"
PLANNING = ROOT / "data" / "results" / "capacity_planning"
LOCKED = ROOT / "data" / "results" / "locked_controller_evaluation"

ORANGE = "#E67E22"
BLUE = "#2980B9"
GREEN = "#27AE60"
GREY = "#6B7280"
LIGHT_GREY = "#D1D5DB"

STEPS_PER_DAY = 96
DT_HOURS = 0.25
DEVICE_LABELS = ("IBVR-17", "IBVR-21", "IBVR-24", "SVC-32")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.labelweight": "bold",
            "axes.linewidth": 0.9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.1,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def finish_axes(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(True, axis=axis)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def export_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        ("png", {"dpi": 400}),
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        fig.savefig(OUT / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def export_figure_fixed_size(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        for suffix, kwargs in (
            ("svg", {}),
            ("pdf", {}),
            ("png", {"dpi": 400}),
            ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
        ):
            fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches=None, **kwargs)
    plt.close(fig)


def framework_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.6), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, colour: str) -> None:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=mpl.colors.to_rgba(colour, 0.10), edgecolor=colour, linewidth=1.8,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", color=colour, fontweight="bold", fontsize=10)
        ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", fontsize=8.5, linespacing=1.25)

    def arrow(start: tuple[float, float], end: tuple[float, float], label: str = "") -> None:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color=GREY))
        if label:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.025, label, ha="center", fontsize=8, color=GREY)

    box(0.04, 0.64, 0.25, 0.25, "Outer capacity planner", "2,000-point capacity mesh\n3/30/200-day staged evaluation\nempirical chance constraints", BLUE)
    box(0.37, 0.64, 0.25, 0.25, "WG-CVaR-PPO", "capacity-conditioned policy\n8 capacity groups\ndirectional tail-risk critics", ORANGE)
    box(0.70, 0.64, 0.25, 0.25, "AC safety projection", "local voltage sensitivities\nminimum action correction\nnonlinear power-flow verification", GREEN)
    box(0.37, 0.16, 0.25, 0.25, "Active distribution network", "IEEE 33-bus AC model\nPV inverters + SVC + capacitors\nvoltage and line-loss feedback", GREY)

    arrow((0.29, 0.77), (0.37, 0.77), r"capacity $\theta$")
    arrow((0.62, 0.77), (0.70, 0.77), "raw action")
    arrow((0.825, 0.64), (0.62, 0.35), "executed action")
    arrow((0.50, 0.41), (0.50, 0.64), "state, loss and voltage")
    arrow((0.37, 0.29), (0.17, 0.64), "empirical risk and energy")
    ax.text(0.5, 0.04, "Policy-aware co-planning: learned efficiency with physics-based safety", ha="center", fontsize=11, fontweight="bold")
    export_figure(fig, "fig_framework")


def locked_generalisation() -> None:
    data = pd.read_csv(LOCKED / "locked_summary.csv")
    projected = data[data["variant"].isin(["wg_projected", "concat_projected"])].copy()
    projected["method"] = projected["variant"].map(
        {"wg_projected": "WG-CVaR + projection", "concat_projected": "Concat + projection"}
    )
    projected.to_csv(OUT / "fig_locked_generalisation_source.csv", index=False)

    modes = ("uniform", "stress")
    mode_labels = {"uniform": "Uniform", "stress": "Stress"}
    reductions: list[float] = []
    per_seed: dict[str, list[float]] = {}
    for mode in modes:
        subset = projected[projected["mode"] == mode].set_index(["variant", "seed"])
        values = []
        for seed in (42, 43, 44):
            wg = float(subset.loc[("wg_projected", seed), "mean_loss_mw"])
            concat = float(subset.loc[("concat_projected", seed), "mean_loss_mw"])
            values.append(100.0 * (concat - wg) / concat)
        per_seed[mode] = values
        wg_mean = projected.query("mode == @mode and variant == 'wg_projected'")["mean_loss_mw"].mean()
        concat_mean = projected.query("mode == @mode and variant == 'concat_projected'")["mean_loss_mw"].mean()
        reductions.append(100.0 * (concat_mean - wg_mean) / concat_mean)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.4, 3.55),
        gridspec_kw={"width_ratios": (1.18, 1.0)},
        constrained_layout=True,
    )

    ax = axes[0]
    seed_markers = ("o", "X", "s")
    seed_offsets = (0.12, 0.0, -0.12)
    row_positions = {"uniform": 1.0, "stress": 0.0}
    for mode, mean_value in zip(modes, reductions):
        y_base = row_positions[mode]
        values = per_seed[mode]
        ax.hlines(y_base, min(values), max(values), color=LIGHT_GREY, linewidth=1.5, zorder=1)
        for seed, marker, offset, value in zip((42, 43, 44), seed_markers, seed_offsets, values):
            colour = "#C0392B" if value < 0 else GREY
            ax.scatter(value, y_base + offset, marker=marker, s=48, color=colour, edgecolor="white", linewidth=0.6, zorder=3)
            ax.annotate(
                str(seed),
                (value, y_base + offset),
                xytext=(0, 7 if offset >= 0 else -10),
                textcoords="offset points",
                ha="center",
                fontsize=6.8,
                color=colour,
            )
        ax.scatter(mean_value, y_base, marker="D", s=78, color=GREEN, edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate(
            f"{mean_value:.2f}% mean",
            (mean_value, y_base),
            xytext=(-9, 0),
            textcoords="offset points",
            va="center",
            ha="right",
            fontsize=7.7,
            fontweight="bold",
            color=GREEN,
        )
    ax.axvline(0, color="#374151", linestyle="--", linewidth=1.1)
    ax.set_yticks((1.0, 0.0), ("Uniform", "Stress"))
    ax.set_xlim(-2.0, 8.2)
    ax.set_ylim(-0.48, 1.48)
    ax.set_xlabel("Line-loss reduction vs projected Concat (%)\nPositive values favour WG-CVaR")
    ax.set_title("Matched line-loss effect")
    ax.grid(True, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.98,
        0.05,
        "Numbers identify policy seeds; diamonds show means",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=GREY,
    )
    panel_label(ax, "(a)")

    ax = axes[1]
    method_specs = (
        ("wg_projected", "WG-CVaR", ORANGE, "o", 0.12),
        ("concat_projected", "Concat", BLUE, "s", -0.12),
    )
    for mode in modes:
        y_base = row_positions[mode]
        values_by_method = []
        for variant, _, _, _, _ in method_specs:
            mean_rate = 100.0 * projected.query("mode == @mode and variant == @variant")["mean_intervention_rate"].mean()
            values_by_method.append(mean_rate)
        ax.hlines(y_base, min(values_by_method), max(values_by_method), color=LIGHT_GREY, linewidth=1.5, zorder=1)
        for (variant, label, colour, marker, offset), mean_rate in zip(method_specs, values_by_method):
            ax.scatter(mean_rate, y_base + offset, marker=marker, s=64, color=colour, edgecolor="white", linewidth=0.7, zorder=3)
            ax.annotate(
                f"{label}  {mean_rate:.5f}%",
                (mean_rate, y_base + offset),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.2,
                color=colour,
            )
    ax.set_xscale("log")
    ax.set_xlim(0.00035, 0.03)
    ax.set_ylim(-0.48, 1.48)
    ax.set_yticks((1.0, 0.0), ("Uniform", "Stress"))
    ax.set_xlabel("Safety intervention rate (%)\nLog scale")
    ax.set_title("Safety-layer burden", pad=21)
    ax.text(
        0.98,
        0.99,
        "0 / 3,000 event-days\nfor every method-set pair",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=GREEN,
        fontsize=7.0,
        fontweight="bold",
    )
    ax.grid(True, axis="x", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_label(ax, "(b)")

    export_figure(fig, "fig_locked_generalisation")


def locked_generalisation_nature_legacy() -> None:
    data = pd.read_csv(LOCKED / "locked_summary.csv")
    projected = data[data["variant"].isin(["wg_projected", "concat_projected"])].copy()
    projected["method"] = projected["variant"].map(
        {"wg_projected": "WG-CVaR", "concat_projected": "Concat"}
    )
    projected.to_csv(OUT / "fig_locked_generalisation_nature_source.csv", index=False)

    blue = "#0F4D92"
    blue_soft = "#AFC6DF"
    neutral = "#767676"
    neutral_dark = "#4D4D4D"
    neutral_light = "#D5D7DA"
    red = "#B64342"
    green = "#2E9E44"

    mpl.rcParams.update(
        {
            "font.size": 7.0,
            "axes.titlesize": 8.3,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.4,
            "axes.labelweight": "normal",
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.3,
            "legend.frameon": False,
        }
    )

    fig = plt.figure(figsize=(7.2, 3.15))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.52, 0.88, 1.08), wspace=0.43)
    ax_loss = fig.add_subplot(grid[0, 0])
    ax_effect = fig.add_subplot(grid[0, 1])
    ax_safety = fig.add_subplot(grid[0, 2])

    # Panel a: pooled absolute performance as grouped bars, with all seeds retained as points.
    group_specs = (("uniform", "Uniform"), ("stress", "Stress"))
    effects: dict[str, list[float]] = {"uniform": [], "stress": []}
    scenario_x = np.arange(2, dtype=float)
    bar_width = 0.32
    concat_means = []
    wg_means = []
    seed_values: dict[str, tuple[list[float], list[float]]] = {}
    for mode, _ in group_specs:
        subset = projected[projected["mode"] == mode].set_index(["variant", "seed"])
        concat_values = []
        wg_values = []
        for seed in (42, 43, 44):
            concat_loss = float(subset.loc[("concat_projected", seed), "mean_loss_mw"])
            wg_loss = float(subset.loc[("wg_projected", seed), "mean_loss_mw"])
            reduction = 100.0 * (concat_loss - wg_loss) / concat_loss
            concat_values.append(concat_loss)
            wg_values.append(wg_loss)
            effects[mode].append(reduction)
        concat_means.append(float(np.mean(concat_values)))
        wg_means.append(float(np.mean(wg_values)))
        seed_values[mode] = (concat_values, wg_values)

    concat_bars = ax_loss.bar(
        scenario_x - bar_width / 2,
        concat_means,
        bar_width,
        color="#B6B8BB",
        edgecolor=neutral,
        linewidth=0.7,
        label="Concat",
        zorder=1,
    )
    wg_bars = ax_loss.bar(
        scenario_x + bar_width / 2,
        wg_means,
        bar_width,
        color=blue,
        edgecolor=blue,
        linewidth=0.7,
        label="WG-CVaR",
        zorder=1,
    )
    seed_jitter = (-0.055, 0.0, 0.055)
    for index, (mode, _) in enumerate(group_specs):
        concat_values, wg_values = seed_values[mode]
        for jitter, concat_loss, wg_loss in zip(seed_jitter, concat_values, wg_values):
            ax_loss.scatter(
                scenario_x[index] - bar_width / 2 + jitter,
                concat_loss,
                s=16,
                facecolor="white",
                edgecolor=neutral_dark,
                linewidth=0.65,
                zorder=3,
            )
            ax_loss.scatter(
                scenario_x[index] + bar_width / 2 + jitter,
                wg_loss,
                s=16,
                facecolor="white",
                edgecolor=blue,
                linewidth=0.65,
                zorder=3,
            )

        mean_reduction = 100.0 * (concat_means[index] - wg_means[index]) / concat_means[index]
        bracket_y = max(concat_values + wg_values) + 0.0016
        x_left = scenario_x[index] - bar_width / 2
        x_right = scenario_x[index] + bar_width / 2
        ax_loss.plot(
            (x_left, x_left, x_right, x_right),
            (bracket_y - 0.00035, bracket_y, bracket_y, bracket_y - 0.00035),
            color=neutral_dark,
            linewidth=0.75,
            zorder=4,
        )
        ax_loss.text(
            scenario_x[index],
            bracket_y + 0.00035,
            f"{mean_reduction:.2f}% lower",
            ha="center",
            va="bottom",
            fontsize=5.9,
            fontweight="bold",
            color=blue,
        )

    for bars, values, text_colour in (
        (concat_bars, concat_means, neutral_dark),
        (wg_bars, wg_means, "white"),
    ):
        for bar, value in zip(bars, values):
            ax_loss.text(
                bar.get_x() + bar.get_width() / 2,
                value * 0.50,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=5.7,
                fontweight="bold",
                color=text_colour,
            )

    ax_loss.set_xticks(scenario_x, ("Uniform", "Stress"))
    ax_loss.set_xlim(-0.50, 1.50)
    ax_loss.set_ylim(0.0, 0.075)
    ax_loss.set_ylabel("Mean line loss (MW)")
    ax_loss.set_title("Matched operating loss", loc="left", pad=6)
    ax_loss.text(-0.13, 1.05, "a", transform=ax_loss.transAxes, fontsize=8.5, fontweight="bold")
    ax_loss.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.105),
        ncol=2,
        fontsize=5.7,
        columnspacing=1.1,
        handlelength=1.25,
    )
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)

    # Panel b: a compact numerical audit of generalisation across seeds and sets.
    pooled_effects = []
    for mode in ("uniform", "stress"):
        subset = projected[projected["mode"] == mode]
        concat_mean = subset[subset["variant"] == "concat_projected"]["mean_loss_mw"].mean()
        wg_mean = subset[subset["variant"] == "wg_projected"]["mean_loss_mw"].mean()
        pooled_effects.append(100.0 * (concat_mean - wg_mean) / concat_mean)
    effect_matrix = np.array(
        [
            [effects["uniform"][0], effects["stress"][0]],
            [effects["uniform"][1], effects["stress"][1]],
            [effects["uniform"][2], effects["stress"][2]],
            pooled_effects,
        ]
    )
    effect_cmap = LinearSegmentedColormap.from_list("wg_effect", (red, "#F7F7F7", blue))
    effect_norm = TwoSlopeNorm(vmin=-1.3, vcenter=0.0, vmax=7.0)
    for row in range(effect_matrix.shape[0]):
        for col in range(effect_matrix.shape[1]):
            value = effect_matrix[row, col]
            ax_effect.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    facecolor=effect_cmap(effect_norm(value)),
                    edgecolor="none",
                )
            )
            text_colour = "white" if value > 4.4 or value < -0.75 else "#202020"
            ax_effect.text(
                col,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                fontweight="bold" if row == 3 else "normal",
                color=text_colour,
            )
    ax_effect.set_xlim(-0.5, 1.5)
    ax_effect.set_ylim(3.5, -0.5)
    ax_effect.add_patch(Rectangle((-0.49, 2.51), 1.98, 0.98, fill=False, edgecolor="#202020", linewidth=1.0))
    ax_effect.set_xticks((0, 1), ("Uniform", "Stress"))
    ax_effect.set_yticks((0, 1, 2, 3), ("Seed 42", "Seed 43", "Seed 44", "Pooled"))
    ax_effect.tick_params(length=0)
    ax_effect.set_title("Line-loss reduction (%)", loc="left", pad=6)
    ax_effect.text(-0.31, 1.05, "b", transform=ax_effect.transAxes, fontsize=8.5, fontweight="bold")
    ax_effect.text(
        0.5,
        -0.13,
        "Blue: WG-CVaR lower   Red: higher",
        transform=ax_effect.transAxes,
        ha="center",
        va="top",
        fontsize=5.2,
        color=neutral_dark,
    )
    for spine in ax_effect.spines.values():
        spine.set_color("#BFC3C7")
        spine.set_linewidth(0.7)

    # Panel c: residual safety and the operating burden of the shared projection.
    ax = ax_safety
    x = np.arange(2)
    width = 0.30
    wg_rates = [
        10000.0 * projected.query("mode == @mode and variant == 'wg_projected'")["mean_intervention_rate"].mean()
        for mode in ("uniform", "stress")
    ]
    concat_rates = [
        10000.0 * projected.query("mode == @mode and variant == 'concat_projected'")["mean_intervention_rate"].mean()
        for mode in ("uniform", "stress")
    ]
    wg_bars = ax.bar(x - width / 2, wg_rates, width, color=blue, label="WG-CVaR")
    concat_bars = ax.bar(x + width / 2, concat_rates, width, color=neutral, label="Concat")
    for bars, values, colour in ((wg_bars, wg_rates, blue), (concat_bars, concat_rates, neutral_dark)):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.045,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=5.8,
                color=colour,
            )
    ax.set_xticks(x, ("Uniform", "Stress"))
    ax.set_ylim(0, 1.82)
    ax.set_ylabel("Projection interventions\nper 10,000 control steps")
    ax.set_title("Safety outcome and burden", loc="left", pad=6)
    ax.text(-0.15, 1.05, "c", transform=ax.transAxes, fontsize=8.5, fontweight="bold")
    ax.text(
        0.50,
        0.98,
        "Zero events / 3,000 days\nfor each method--set pair",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.55,
        fontweight="bold",
        color=green,
    )
    ax.text(
        1.0,
        1.56,
        f"{wg_rates[1] / concat_rates[1]:.2f}x intervention burden",
        ha="center",
        va="bottom",
        fontsize=5.3,
        color=neutral_dark,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), fontsize=5.8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.078, right=0.992, bottom=0.20, top=0.89)
    export_figure_fixed_size(fig, "fig_locked_generalisation_nature")


def locked_generalisation_nature() -> None:
    """Render the locked controller comparison in the manuscript's engineering style."""
    data = pd.read_csv(LOCKED / "locked_summary.csv")
    projected = data[data["variant"].isin(["wg_projected", "concat_projected"])].copy()
    projected["method"] = projected["variant"].map(
        {"wg_projected": "WG-CVaR", "concat_projected": "Concat"}
    )
    projected.to_csv(OUT / "fig_locked_generalisation_nature_source.csv", index=False)

    modes = ("uniform", "stress")
    mode_labels = ("Uniform", "Stress")
    seeds = (42, 43, 44)
    seed_colours = {42: "#443983", 43: "#21918C", 44: "#90D743"}
    seed_markers = {42: "o", 43: "s", 44: "^"}
    dark = "#303030"
    red = "#C0392B"

    mean_loss: dict[str, list[float]] = {"concat_projected": [], "wg_projected": []}
    seed_loss: dict[str, dict[str, list[float]]] = {}
    reductions: dict[int | str, list[float]] = {seed: [] for seed in seeds}
    reductions["Pooled"] = []
    interventions: dict[str, list[float]] = {"concat_projected": [], "wg_projected": []}

    for mode in modes:
        subset = projected[projected["mode"] == mode].set_index(["variant", "seed"])
        seed_loss[mode] = {"concat_projected": [], "wg_projected": []}
        for variant in ("concat_projected", "wg_projected"):
            values = [float(subset.loc[(variant, seed), "mean_loss_mw"]) for seed in seeds]
            seed_loss[mode][variant] = values
            mean_loss[variant].append(float(np.mean(values)))
            rate = projected.query("mode == @mode and variant == @variant")["mean_intervention_rate"].mean()
            interventions[variant].append(10000.0 * float(rate))
        for index, seed in enumerate(seeds):
            concat_value = seed_loss[mode]["concat_projected"][index]
            wg_value = seed_loss[mode]["wg_projected"][index]
            reductions[seed].append(100.0 * (concat_value - wg_value) / concat_value)
        concat_mean = mean_loss["concat_projected"][-1]
        wg_mean = mean_loss["wg_projected"][-1]
        reductions["Pooled"].append(100.0 * (concat_mean - wg_mean) / concat_mean)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 1.1,
            "legend.fontsize": 8.0,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
        }
    )

    fig = plt.figure(figsize=(8.5, 3.75))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.55, 1.25, 1.15), wspace=0.38)
    ax_loss = fig.add_subplot(grid[0, 0])
    seed_grid = grid[0, 1].subgridspec(
        2,
        1,
        height_ratios=(6.0, 1.0),
        hspace=0.72,
    )
    ax_seed = fig.add_subplot(seed_grid[0, 0])
    ax_seed_colourbar = fig.add_subplot(seed_grid[1, 0])
    ax_effort = fig.add_subplot(grid[0, 2])

    # (a) Absolute locked-set operating loss with seed-level observations.
    group_x = np.arange(2, dtype=float)
    width = 0.32
    concat_x = group_x - width / 2
    wg_x = group_x + width / 2
    concat_bars = ax_loss.bar(
        concat_x,
        mean_loss["concat_projected"],
        width,
        color=BLUE,
        edgecolor=dark,
        linewidth=0.8,
        alpha=0.90,
        label="Concat",
        zorder=2,
    )
    wg_bars = ax_loss.bar(
        wg_x,
        mean_loss["wg_projected"],
        width,
        color=ORANGE,
        edgecolor=dark,
        linewidth=0.8,
        alpha=0.90,
        label="WG-CVaR",
        zorder=2,
    )
    for mode_index, mode in enumerate(modes):
        concat_mean = mean_loss["concat_projected"][mode_index]
        wg_mean = mean_loss["wg_projected"][mode_index]
        effect = reductions["Pooled"][mode_index]
        ax_loss.add_patch(
            FancyArrowPatch(
                (concat_x[mode_index], concat_mean),
                (wg_x[mode_index], wg_mean),
                arrowstyle="-|>",
                mutation_scale=8,
                linestyle="--",
                linewidth=1.0,
                color=dark,
                zorder=5,
            )
        )
        ax_loss.text(
            group_x[mode_index],
            max(concat_mean, wg_mean) + 0.0045,
            f"{effect:.2f}% lower",
            ha="center",
            va="center",
            fontsize=8.1,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#808080", "linewidth": 0.8},
            zorder=6,
        )

    for bars, values in (
        (concat_bars, mean_loss["concat_projected"]),
        (wg_bars, mean_loss["wg_projected"]),
    ):
        for bar, value in zip(bars, values):
            ax_loss.text(
                bar.get_x() + bar.get_width() / 2,
                0.030,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=7.8,
                fontweight="bold",
                color="white",
            )
    ax_loss.set_xticks(group_x, mode_labels)
    ax_loss.set_ylim(-0.010, 0.075)
    ax_loss.set_yticks(np.arange(0.0, 0.071, 0.01))
    ax_loss.set_ylabel("Mean line loss (MW)")
    ax_loss.set_title("(a) Matched operating loss", loc="left", pad=8)
    ax_loss.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        facecolor="white",
        edgecolor="#A0A0A0",
        borderpad=0.4,
        columnspacing=0.9,
    )
    ax_loss.set_axisbelow(True)
    ax_loss.grid(axis="y", color="#BEBEBE")

    # (b) Eight-cell audit: three seeds plus the pooled effect on two scenario sets.
    effect_matrix = np.array(
        [
            reductions[42],
            reductions[43],
            reductions[44],
            reductions["Pooled"],
        ]
    )
    effect_cmap = LinearSegmentedColormap.from_list(
        "locked_effect",
        (red, "#F7F7F7", BLUE),
    )
    effect_norm = TwoSlopeNorm(vmin=-1.3, vcenter=0.0, vmax=7.0)
    for row in range(effect_matrix.shape[0]):
        for col in range(effect_matrix.shape[1]):
            value = effect_matrix[row, col]
            ax_seed.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    facecolor=effect_cmap(effect_norm(value)),
                    edgecolor="white",
                    linewidth=0.7,
                )
            )
            text_colour = "white" if value > 4.5 or value < -0.75 else dark
            ax_seed.text(
                col,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=9.0,
                fontweight="bold" if row == 3 else "normal",
                color=text_colour,
            )
    # A separator identifies the pooled result without implying a confidence box.
    ax_seed.axhline(2.5, color=dark, linewidth=1.15)
    ax_seed.set_xlim(-0.5, 1.5)
    ax_seed.set_ylim(3.5, -0.5)
    ax_seed.set_xticks((0, 1), mode_labels)
    ax_seed.set_yticks((0, 1, 2, 3), ("Seed 42", "Seed 43", "Seed 44", "Pooled"))
    ax_seed.tick_params(axis="x", length=0, pad=5)
    ax_seed.tick_params(axis="y", length=0)
    ax_seed.set_title("(b) Seed-level reduction", loc="left", pad=8)
    for spine in ax_seed.spines.values():
        spine.set_color(dark)
        spine.set_linewidth(1.1)

    colour_bar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=effect_norm, cmap=effect_cmap),
        cax=ax_seed_colourbar,
        orientation="horizontal",
    )
    colour_bar.set_ticks((-1.0, 0.0, 3.0, 7.0))
    colour_bar.ax.tick_params(labelsize=7.0, length=2.2, pad=1.5)
    colour_bar.outline.set_linewidth(0.7)
    colour_bar.ax.text(
        0.0,
        1.55,
        "Concat lower",
        transform=colour_bar.ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        fontweight="bold",
        color=dark,
    )
    colour_bar.ax.text(
        1.0,
        1.55,
        "WG-CVaR lower",
        transform=colour_bar.ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        fontweight="bold",
        color=dark,
    )
    colour_bar.set_label("Relative line-loss reduction (%)", fontsize=7.0, labelpad=1.5)

    # (c) Projection effort at the same zero-event observed safety outcome.
    effort_width = 0.32
    concat_effort_x = group_x - effort_width / 2
    wg_effort_x = group_x + effort_width / 2
    concat_effort = interventions["concat_projected"]
    wg_effort = interventions["wg_projected"]
    concat_effort_bars = ax_effort.bar(
        concat_effort_x,
        concat_effort,
        effort_width,
        color=BLUE,
        edgecolor=dark,
        linewidth=0.8,
        alpha=0.90,
        zorder=2,
    )
    wg_effort_bars = ax_effort.bar(
        wg_effort_x,
        wg_effort,
        effort_width,
        color=ORANGE,
        edgecolor=dark,
        linewidth=0.8,
        alpha=0.90,
        zorder=2,
    )
    for bars, values in ((concat_effort_bars, concat_effort), (wg_effort_bars, wg_effort)):
        for bar, value in zip(bars, values):
            ax_effort.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.045,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
                color=dark,
            )
    stress_ratio = wg_effort[1] / concat_effort[1]
    ax_effort.add_patch(
        FancyArrowPatch(
            (concat_effort_x[1], concat_effort[1]),
            (wg_effort_x[1], wg_effort[1]),
            arrowstyle="-|>",
            mutation_scale=8,
            linestyle="--",
            linewidth=1.0,
            color=dark,
            zorder=4,
        )
    )
    ax_effort.text(
        group_x[1],
        (concat_effort[1] + wg_effort[1]) / 2 + 0.06,
        f"{stress_ratio:.2f}x higher",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#808080", "linewidth": 0.8},
        zorder=5,
    )
    ax_effort.text(
        0.50,
        0.97,
        "0 / 3,000 event-days\nfor every method-set pair",
        transform=ax_effort.transAxes,
        ha="center",
        va="top",
        fontsize=7.8,
        fontweight="normal",
        color=dark,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#808080", "linewidth": 0.8},
    )
    ax_effort.set_xticks(group_x, mode_labels)
    ax_effort.set_ylim(0.0, 1.90)
    ax_effort.set_ylabel("Interventions per 10,000 steps")
    ax_effort.set_title("(c) Projection effort", loc="left", pad=8)
    ax_effort.set_axisbelow(True)
    ax_effort.grid(axis="y", color="#BEBEBE")

    fig.subplots_adjust(left=0.073, right=0.992, bottom=0.16, top=0.88)
    export_figure_fixed_size(fig, "fig_locked_generalisation_nature")


def projection_ablation() -> None:
    data = pd.read_csv(PLANNING / "ablation" / "comparison.csv")
    data = data[data["candidate_id"] == 0].copy()
    order = ("wg_raw", "wg_projected", "concat_raw", "concat_projected")
    data = data.set_index("variant").loc[list(order)].reset_index()
    data.to_csv(OUT / "fig_projection_ablation_source.csv", index=False)

    labels = ("WG raw", "WG + proj.", "Concat raw", "Concat + proj.")
    colours = (ORANGE, ORANGE, BLUE, BLUE)
    hatches = ("//", "", "//", "")
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.05), constrained_layout=True)

    metrics = (
        (100.0 * data["worst_seed_risk"], "Worst-seed event-day rate (%)", "Residual policy risk"),
        (data["mean_daily_loss_mwh"], "Mean daily line loss (MWh)", "Operating efficiency"),
        (100.0 * data["mean_intervention_rate"], "Intervened control steps (%)", "Projection effort"),
    )
    for index, (ax, (values, ylabel, title)) in enumerate(zip(axes, metrics)):
        bars = ax.bar(np.arange(4), values, color=colours, edgecolor=[ORANGE, ORANGE, BLUE, BLUE], width=0.68)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_xticks(np.arange(4), labels, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if index == 0:
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.0f}%", ha="center", fontweight="bold")
        finish_axes(ax)
        panel_label(ax, f"({chr(97 + index)})")
    export_figure(fig, "fig_projection_ablation")


def projection_ablation_nature() -> None:
    data = pd.read_csv(PLANNING / "ablation" / "comparison.csv")
    data = data[data["candidate_id"] == 0].set_index("variant")

    blue = "#0F4D92"
    neutral = "#777777"
    dark = "#252525"

    mpl.rcParams.update(
        {
            "font.size": 7.0,
            "axes.titlesize": 8.3,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.4,
            "axes.labelweight": "normal",
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    fig = plt.figure(figsize=(7.2, 2.55))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.24, 1.04, 1.0), wspace=0.48)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]

    x = np.arange(2)
    method_labels = ("WG-CVaR", "Concat")
    method_colours = (blue, neutral)

    def academic_axes(ax: plt.Axes) -> None:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E2E4E7", linewidth=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def difference_bracket(ax: plt.Axes, y_value: float, height: float, text_value: str) -> None:
        ax.plot((0, 0, 1, 1), (y_value - height, y_value, y_value, y_value - height), color=dark, linewidth=0.75)
        ax.text(0.5, y_value + height * 0.55, text_value, ha="center", va="bottom", fontsize=5.8, fontweight="bold", color=dark)

    # Panel a: conventional paired dot plot for the raw-to-projected risk change.
    ax = axes[0]
    raw_risk = 100.0 * np.array([float(data.loc["wg_raw", "worst_seed_risk"]), float(data.loc["concat_raw", "worst_seed_risk"])])
    projected_risk = 100.0 * np.array([float(data.loc["wg_projected", "worst_seed_risk"]), float(data.loc["concat_projected", "worst_seed_risk"])])
    for index, colour in enumerate(method_colours):
        ax.plot((index, index), (projected_risk[index], raw_risk[index]), color=colour, linewidth=1.4, zorder=1)
        ax.scatter(index, raw_risk[index], s=38, facecolor="white", edgecolor=colour, linewidth=1.15, zorder=3)
        ax.scatter(index, projected_risk[index], marker="s", s=34, facecolor=colour, edgecolor="white", linewidth=0.55, zorder=4)
        ax.text(index, raw_risk[index] + 1.0, f"{raw_risk[index]:.0f}%", ha="center", va="bottom", fontsize=6.0, fontweight="bold", color=colour)
        ax.text(index + 0.06, projected_risk[index] + 0.7, "0%", ha="left", va="bottom", fontsize=5.8, fontweight="bold", color=colour)
    ax.scatter([], [], s=34, facecolor="white", edgecolor=dark, linewidth=1.0, label="Raw policy")
    ax.scatter([], [], marker="s", s=30, facecolor=dark, edgecolor="white", linewidth=0.5, label="With projection")
    ax.set_xticks(x, method_labels)
    ax.set_ylim(-1.5, 30.0)
    ax.set_ylabel("Worst-seed event-day rate (%)")
    ax.set_title("Worst-seed empirical risk", loc="left", pad=6)
    ax.text(-0.15, 1.05, "a", transform=ax.transAxes, fontsize=8.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=5.5, handletextpad=0.4, borderaxespad=0.2)
    academic_axes(ax)

    # Panel b: projected-policy loss at the same zero-observed-event outcome.
    ax = axes[1]
    projected_losses = np.array([float(data.loc["wg_projected", "mean_daily_loss_mwh"]), float(data.loc["concat_projected", "mean_daily_loss_mwh"])])
    bars = ax.bar(x, projected_losses, width=0.58, color=method_colours, edgecolor="none")
    for bar, value, colour in zip(bars, projected_losses, method_colours):
        ax.text(bar.get_x() + bar.get_width() / 2, value - 0.055, f"{value:.4f}", ha="center", va="center", fontsize=5.9, fontweight="bold", color="white")
    loss_gain = 100.0 * (projected_losses[1] - projected_losses[0]) / projected_losses[1]
    difference_bracket(ax, 0.965, 0.012, f"{loss_gain:.2f}% lower")
    ax.set_xticks(x, method_labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Mean daily line-loss energy (MWh)")
    ax.set_title("Projected-policy line loss", loc="left", pad=6)
    ax.text(-0.15, 1.05, "b", transform=ax.transAxes, fontsize=8.5, fontweight="bold")
    academic_axes(ax)

    # Panel c: projected-policy intervention burden only.
    ax = axes[2]
    projected_effort = 100.0 * np.array([float(data.loc["wg_projected", "mean_intervention_rate"]), float(data.loc["concat_projected", "mean_intervention_rate"])])
    bars = ax.bar(x, projected_effort, width=0.58, color=method_colours, edgecolor="none")
    for bar, value in zip(bars, projected_effort):
        ax.text(bar.get_x() + bar.get_width() / 2, value - 0.018, f"{value:.3f}%", ha="center", va="center", fontsize=5.9, fontweight="bold", color="white")
    effort_ratio = projected_effort[0] / projected_effort[1]
    difference_bracket(ax, 0.260, 0.006, f"{effort_ratio:.2f}x")
    ax.set_xticks(x, method_labels)
    ax.set_ylim(0.0, 0.278)
    ax.set_ylabel("Intervened control steps (%)")
    ax.set_title("Projection intervention rate", loc="left", pad=6)
    ax.text(-0.15, 1.05, "c", transform=ax.transAxes, fontsize=8.5, fontweight="bold")
    academic_axes(ax)

    fig.subplots_adjust(left=0.088, right=0.992, bottom=0.22, top=0.87)
    export_figure_fixed_size(fig, "fig_projection_ablation_nature")


def projection_ablation_distribution() -> None:
    base = PLANNING / "ablation"
    summary = pd.read_csv(base / "comparison.csv")
    summary = summary[summary["candidate_id"] == 0].set_index("variant")
    variants = ("wg_raw", "wg_projected", "concat_raw", "concat_projected")
    variant_labels = ("Raw", "Projected", "Raw", "Projected")
    positions = np.array((0.0, 0.9, 2.25, 3.15))
    seed_markers = {42: "o", 43: "s", 44: "^"}
    seed_colours = {42: "#443983", 43: "#21918C", 44: "#90D743"}
    method_colours = (ORANGE, ORANGE, BLUE, BLUE)
    red = "#C0392B"
    dark = "#303030"

    frames = []
    for variant in variants:
        for seed in (42, 43, 44):
            frame = pd.read_csv(base / variant / f"seed{seed}" / "episodes.csv")
            frame = frame[frame["candidate_id"] == 0].copy()
            frame["variant"] = variant
            frame["seed"] = seed
            frames.append(frame)
    daily = pd.concat(frames, ignore_index=True)
    daily.to_csv(OUT / "fig_projection_ablation_distribution_source.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 10.0,
            "axes.titlesize": 12.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.3,
            "ytick.labelsize": 9.3,
            "axes.linewidth": 1.15,
            "legend.fontsize": 8.8,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
        }
    )

    fig = plt.figure(figsize=(8.5, 3.65))
    grid = fig.add_gridspec(1, 2, width_ratios=(2.65, 1.0), wspace=0.30)
    ax = fig.add_subplot(grid[0, 0])
    ax_effort = fig.add_subplot(grid[0, 1])
    rng = np.random.default_rng(20260720)

    distributions = []
    for position, variant in zip(positions, variants):
        variant_data = daily[daily["variant"] == variant]
        distributions.append(variant_data["daily_loss_mwh"].to_numpy())
        for seed in (42, 43, 44):
            seed_data = variant_data[variant_data["seed"] == seed]
            jitter = rng.normal(0.0, 0.050, len(seed_data))
            x_values = position + jitter
            ax.scatter(
                x_values,
                seed_data["daily_loss_mwh"],
                s=12,
                marker=seed_markers[seed],
                color=seed_colours[seed],
                alpha=0.34,
                edgecolor="none",
                rasterized=False,
                zorder=1,
            )
            event_mask = seed_data["risk"].to_numpy(dtype=float) > 0.0
            if np.any(event_mask):
                ax.scatter(
                    x_values[event_mask],
                    seed_data.loc[event_mask, "daily_loss_mwh"],
                    s=20,
                    marker="x",
                    color=red,
                    alpha=0.75,
                    linewidth=0.85,
                    zorder=2,
                )

    box = ax.boxplot(
        distributions,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": dark, "linewidth": 1.65},
        boxprops={"facecolor": "white", "edgecolor": dark, "linewidth": 1.35},
        whiskerprops={"color": dark, "linewidth": 1.15},
        capprops={"color": dark, "linewidth": 1.15},
    )
    for patch, colour in zip(box["boxes"], method_colours):
        patch.set_edgecolor(colour)
        patch.set_facecolor(colour)
        patch.set_alpha(0.20)

    all_losses = daily["daily_loss_mwh"].to_numpy()
    y_min = float(np.floor((all_losses.min() - 0.038) * 100.0) / 100.0)
    y_max = float(np.ceil((all_losses.max() + 0.075) * 100.0) / 100.0)

    wg_raw_risk = 100.0 * float(summary.loc["wg_raw", "worst_seed_risk"])
    wg_projected_risk = 100.0 * float(summary.loc["wg_projected", "worst_seed_risk"])
    concat_raw_risk = 100.0 * float(summary.loc["concat_raw", "worst_seed_risk"])
    concat_projected_risk = 100.0 * float(summary.loc["concat_projected", "worst_seed_risk"])
    risk_y = y_max - 0.007
    ax.text(
        positions[:2].mean(),
        risk_y,
        f"Worst-seed risk: {wg_raw_risk:.1f}% $\\rightarrow$ {wg_projected_risk:.0f}%",
        ha="center",
        va="top",
        fontsize=8.7,
        fontweight="bold",
        color=ORANGE,
    )
    ax.text(
        positions[2:].mean(),
        risk_y,
        f"Worst-seed risk: {concat_raw_risk:.0f}% $\\rightarrow$ {concat_projected_risk:.0f}%",
        ha="center",
        va="top",
        fontsize=8.7,
        fontweight="bold",
        color=BLUE,
    )

    wg_projected_mean = float(summary.loc["wg_projected", "mean_daily_loss_mwh"])
    concat_projected_mean = float(summary.loc["concat_projected", "mean_daily_loss_mwh"])
    projected_gain = 100.0 * (concat_projected_mean - wg_projected_mean) / concat_projected_mean
    ax.scatter(
        (positions[1], positions[3]),
        (wg_projected_mean, concat_projected_mean),
        marker="D",
        s=34,
        facecolor="white",
        edgecolor=dark,
        linewidth=1.1,
        zorder=5,
    )
    ax.add_patch(
        FancyArrowPatch(
            (positions[3], concat_projected_mean),
            (positions[1], wg_projected_mean),
            arrowstyle="-|>",
            mutation_scale=10,
            linestyle="--",
            linewidth=1.25,
            color=dark,
            zorder=4,
        )
    )
    ax.text(
        (positions[1] + positions[3]) / 2,
        (wg_projected_mean + concat_projected_mean) / 2 + 0.008,
        f"{projected_gain:.2f}% lower\nprojected mean loss",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=dark,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#808080", "linewidth": 0.9},
        zorder=6,
    )

    ax.axvline(1.575, color="#B8B8B8", linestyle="--", linewidth=1.0)
    ax.set_xticks(positions, variant_labels)
    ax.set_xlim(-0.45, 3.60)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Daily line-loss energy (MWh)")
    ax.set_title("(a) Daily operating-loss distribution", loc="left", pad=9)
    ax.text(positions[:2].mean(), y_min - 0.027, "WG-CVaR", ha="center", va="top", fontsize=9.5, fontweight="bold", color=ORANGE, clip_on=False)
    ax.text(positions[2:].mean(), y_min - 0.027, "Concat", ha="center", va="top", fontsize=9.5, fontweight="bold", color=BLUE, clip_on=False)
    for seed in (42, 43, 44):
        ax.scatter(
            [],
            [],
            s=24,
            marker=seed_markers[seed],
            facecolor=seed_colours[seed],
            edgecolor="white",
            linewidth=0.45,
            label=f"Seed {seed}",
        )
    ax.scatter([], [], s=24, marker="x", color=red, linewidth=0.9, label="Voltage-event day")
    ax.scatter([], [], s=30, marker="D", facecolor="white", edgecolor=dark, linewidth=1.0, label="Mean")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        handletextpad=0.35,
        columnspacing=0.9,
        borderpad=0.45,
        facecolor="white",
        edgecolor="#A0A0A0",
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#BEBEBE")

    projected_effort = 100.0 * np.array(
        [float(summary.loc["wg_projected", "mean_intervention_rate"]), float(summary.loc["concat_projected", "mean_intervention_rate"])]
    )
    effort_bars = ax_effort.bar(
        np.arange(2), projected_effort, width=0.58, color=(ORANGE, BLUE), edgecolor=dark, linewidth=0.8, alpha=0.90
    )
    for bar, value in zip(effort_bars, projected_effort):
        ax_effort.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.006,
            f"{value:.3f}%",
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
            color=dark,
        )
    ratio = projected_effort[0] / projected_effort[1]
    ax_effort.plot((0, 0, 1, 1), (0.262, 0.270, 0.270, 0.262), color=dark, linewidth=1.0)
    ax_effort.text(0.5, 0.274, f"{ratio:.2f}x", ha="center", va="bottom", fontsize=9.0, fontweight="bold")
    ax_effort.set_xticks((0, 1), ("WG-CVaR", "Concat"))
    ax_effort.set_ylim(0.0, 0.292)
    ax_effort.set_ylabel("Intervened control steps (%)")
    ax_effort.set_title("(b) Projection effort", loc="left", pad=9)
    ax_effort.set_axisbelow(True)
    ax_effort.grid(axis="y", color="#BEBEBE")

    fig.subplots_adjust(left=0.085, right=0.987, bottom=0.20, top=0.88)
    export_figure_fixed_size(fig, "fig_projection_ablation_distribution")


def capacity_planning() -> None:
    raw = pd.read_csv(PLANNING / "risk_path_raw" / "wg_raw_candidate_summary.csv")
    projected_all = pd.read_csv(PLANNING / "confirm" / "wg_projected_candidate_summary.csv")
    projected = projected_all[projected_all["selection_role"] == "risk_path"].copy()
    path = raw.merge(
        projected[
            ["candidate_id", "worst_seed_risk", "mean_intervention_rate", "mean_daily_loss_mwh"]
        ],
        on="candidate_id",
        suffixes=("_raw", "_projected"),
    ).sort_values("resource_index")
    path.to_csv(OUT / "fig_capacity_planning_source.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 1.1,
            "legend.fontsize": 8.0,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
        }
    )

    resource = path["resource_index"].to_numpy(dtype=float)
    raw_risk = 100.0 * path["worst_seed_risk_raw"].to_numpy(dtype=float)
    projected_risk = 100.0 * path["worst_seed_risk_projected"].to_numpy(dtype=float)
    intervention_rate = 100.0 * path["mean_intervention_rate_projected"].to_numpy(dtype=float)
    projected_loss = path["mean_daily_loss_mwh_projected"].to_numpy(dtype=float)

    raw_feasible_index = int(np.flatnonzero(raw_risk <= 1.0)[0])
    projected_feasible_index = int(np.flatnonzero(projected_risk <= 1.0)[0])
    raw_feasible_resource = float(resource[raw_feasible_index])
    projected_feasible_resource = float(resource[projected_feasible_index])
    resource_reduction = 100.0 * (
        raw_feasible_resource - projected_feasible_resource
    ) / raw_feasible_resource
    endpoint_loss_reduction = 100.0 * (projected_loss[-1] - projected_loss[0]) / projected_loss[-1]

    dark = "#303030"
    raw_colour = "#6B7280"
    threshold_colour = "#B94A48"
    safe_fill = "#E8F3EA"

    fig = plt.figure(figsize=(8.5, 3.75))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.45, 1.12, 1.38), wspace=0.39)
    ax_risk = fig.add_subplot(grid[0, 0])
    ax_effort = fig.add_subplot(grid[0, 1])
    ax_loss = fig.add_subplot(grid[0, 2])

    # (a) The physical projection shifts the path-specific empirical boundary.
    ax_risk.axhspan(0.0, 1.0, color=safe_fill, zorder=0)
    ax_risk.plot(
        resource,
        raw_risk,
        linestyle="--",
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=0.9,
        color=raw_colour,
        linewidth=1.45,
        label="Raw WG-CVaR",
        zorder=3,
    )
    ax_risk.plot(
        resource,
        projected_risk,
        linestyle="-",
        marker="o",
        markersize=3.8,
        color=ORANGE,
        linewidth=1.65,
        label="WG-CVaR + projection",
        zorder=4,
    )
    ax_risk.axhline(
        1.0,
        color=threshold_colour,
        linestyle=(0, (4, 3)),
        linewidth=1.05,
        label="1% empirical threshold",
        zorder=2,
    )
    ax_risk.scatter(
        [projected_feasible_resource],
        [projected_risk[projected_feasible_index]],
        marker="D",
        s=48,
        color=ORANGE,
        edgecolor=dark,
        linewidth=0.8,
        zorder=6,
    )
    ax_risk.scatter(
        [raw_feasible_resource],
        [raw_risk[raw_feasible_index]],
        marker="D",
        s=48,
        facecolor="white",
        edgecolor=raw_colour,
        linewidth=1.2,
        zorder=6,
    )
    boundary_y = 19.0
    ax_risk.add_patch(
        FancyArrowPatch(
            (raw_feasible_resource, boundary_y),
            (projected_feasible_resource, boundary_y),
            arrowstyle="-|>",
            mutation_scale=9,
            linestyle="--",
            linewidth=1.0,
            color=dark,
            zorder=5,
        )
    )
    ax_risk.text(
        86.0,
        boundary_y + 1.4,
        f"{resource_reduction:.2f}% lower\nresource boundary",
        ha="center",
        va="bottom",
        fontsize=7.7,
        fontweight="bold",
        color=dark,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#808080", "linewidth": 0.8},
    )
    ax_risk.annotate(
        "Raw feasible",
        xy=(raw_feasible_resource, raw_risk[raw_feasible_index]),
        xytext=(96.0, 5.2),
        fontsize=7.2,
        color=raw_colour,
        arrowprops={"arrowstyle": "->", "color": raw_colour, "linewidth": 0.8},
    )
    ax_risk.set_xlim(67.0, 155.0)
    ax_risk.set_ylim(-0.8, 28.5)
    ax_risk.set_xticks((70, 90, 110, 130, 150))
    ax_risk.set_ylabel("Worst-seed event-day rate (%)")
    ax_risk.set_title("(a) Path-specific risk boundary", loc="left", pad=8)
    ax_risk.legend(
        loc="center right",
        bbox_to_anchor=(0.98, 0.58),
        facecolor="white",
        edgecolor="#A0A0A0",
        borderpad=0.4,
    )
    ax_risk.set_axisbelow(True)
    ax_risk.grid(axis="y", color="#BEBEBE")

    # (b) Intervention is concentrated at the low-resource end of the path.
    ax_effort.axvspan(
        resource[0],
        raw_feasible_resource,
        color=ORANGE,
        alpha=0.065,
        linewidth=0,
        zorder=0,
    )
    ax_effort.fill_between(resource, 0.0, intervention_rate, color=ORANGE, alpha=0.20, zorder=1)
    ax_effort.plot(
        resource,
        intervention_rate,
        "-o",
        color=ORANGE,
        markeredgecolor=dark,
        markeredgewidth=0.55,
        markersize=4.0,
        linewidth=1.6,
        zorder=3,
    )
    ax_effort.text(
        resource[0] + 1.5,
        intervention_rate[0] + 0.009,
        f"{intervention_rate[0]:.3f}%",
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=dark,
    )
    ax_effort.annotate(
        "No observed intervention\nfrom index 90.5",
        xy=(resource[4], intervention_rate[4]),
        xytext=(105.0, 0.080),
        ha="left",
        va="center",
        fontsize=7.3,
        color=dark,
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#909090", "linewidth": 0.7},
        arrowprops={"arrowstyle": "->", "color": dark, "linewidth": 0.8},
    )
    ax_effort.set_xlim(67.0, 155.0)
    ax_effort.set_ylim(0.0, 0.265)
    ax_effort.set_xticks((70, 90, 110, 130, 150))
    ax_effort.set_ylabel("Intervened control steps (%)")
    ax_effort.set_title("(b) Projection burden", loc="left", pad=8)
    ax_effort.set_axisbelow(True)
    ax_effort.grid(axis="y", color="#BEBEBE")

    # (c) The predefined high-redundancy endpoint carries a substantial loss penalty.
    ax_loss.plot(
        resource,
        projected_loss,
        "-o",
        color=ORANGE,
        markeredgecolor=dark,
        markeredgewidth=0.55,
        markersize=4.0,
        linewidth=1.6,
        zorder=3,
    )
    ax_loss.scatter(
        [resource[0]],
        [projected_loss[0]],
        marker="D",
        s=58,
        color=GREEN,
        edgecolor=dark,
        linewidth=0.8,
        zorder=5,
    )
    ax_loss.scatter(
        [resource[-1]],
        [projected_loss[-1]],
        marker="D",
        s=58,
        color=raw_colour,
        edgecolor=dark,
        linewidth=0.8,
        zorder=5,
    )
    ax_loss.add_patch(
        FancyArrowPatch(
            (resource[-1], projected_loss[-1]),
            (resource[0], projected_loss[0]),
            arrowstyle="-|>",
            mutation_scale=9,
            linestyle="--",
            linewidth=1.05,
            color=dark,
            zorder=4,
        )
    )
    ax_loss.text(
        104.0,
        1.65,
        f"{endpoint_loss_reduction:.2f}% lower\ndaily line loss",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=dark,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#808080", "linewidth": 0.8},
        zorder=6,
    )
    ax_loss.annotate(
        "Minimum resource",
        xy=(resource[0], projected_loss[0]),
        xytext=(94.0, projected_loss[0]),
        ha="left",
        va="center",
        fontsize=7.3,
        color="#24743D",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": GREEN, "linewidth": 0.7},
        arrowprops={"arrowstyle": "-", "color": GREEN, "linewidth": 0.9},
    )
    ax_loss.annotate(
        "High-redundancy\nreference",
        xy=(resource[-1], projected_loss[-1]),
        xytext=(111.0, projected_loss[-1]),
        ha="left",
        va="center",
        fontsize=7.3,
        color=raw_colour,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": raw_colour, "linewidth": 0.7},
        arrowprops={"arrowstyle": "-", "color": raw_colour, "linewidth": 0.9},
    )
    ax_loss.set_xlim(67.0, 155.0)
    ax_loss.set_ylim(0.75, 2.48)
    ax_loss.set_xticks((70, 90, 110, 130, 150))
    ax_loss.set_ylabel("Mean daily line loss (MWh)")
    ax_loss.set_title("(c) Operating loss along path", loc="left", pad=8)
    ax_loss.set_axisbelow(True)
    ax_loss.grid(axis="y", color="#BEBEBE")

    fig.supxlabel("Normalised resource index", fontsize=10.0, fontweight="bold", y=0.045)
    fig.subplots_adjust(left=0.078, right=0.992, bottom=0.20, top=0.89)
    export_figure_fixed_size(fig, "fig_capacity_planning")


def resource_efficiency() -> None:
    data = pd.read_csv(PLANNING / "confirm" / "wg_projected_candidate_summary.csv")
    data["display_role"] = data["selection_role"].map(
        {"optimiser_candidate": "Optimiser candidate", "risk_path": "Predefined capacity path"}
    )
    data["highlight"] = ""
    data.loc[data["candidate_id"] == 0, "highlight"] = "Minimum resource"
    data.loc[data["candidate_id"] == 400, "highlight"] = "Pareto alternative"
    data.loc[data["candidate_id"] == 2016, "highlight"] = "High redundancy"
    data.to_csv(OUT / "fig_resource_efficiency_source.csv", index=False)

    optimiser = data[data["selection_role"] == "optimiser_candidate"].copy()
    path = data[data["selection_role"] == "risk_path"].sort_values("resource_index")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = mpl.colormaps["viridis"]

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 1.1,
            "legend.fontsize": 8.0,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
        }
    )

    dark = "#303030"
    pareto_colour = BLUE
    minimum = data[data["candidate_id"] == 0].iloc[0]
    pareto = data[data["candidate_id"] == 400].iloc[0]
    high = data[data["candidate_id"] == 2016].iloc[0]
    pareto_loss_change = 100.0 * (
        float(pareto["mean_daily_loss_mwh"]) - float(minimum["mean_daily_loss_mwh"])
    ) / float(minimum["mean_daily_loss_mwh"])

    fig = plt.figure(figsize=(8.5, 4.05))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.62, 1.05, 0.055), wspace=0.34)
    ax_global = fig.add_subplot(grid[0, 0])
    ax_low = fig.add_subplot(grid[0, 1])
    ax_colourbar = fig.add_subplot(grid[0, 2])

    # (a) Confirmation landscape across all retained records.
    ax_global.plot(
        path["resource_index"],
        path["mean_daily_loss_mwh"],
        color="#7B8491",
        linewidth=1.15,
        linestyle="--",
        alpha=0.90,
        label="Capacity path",
        zorder=1,
    )
    optimiser_scatter = ax_global.scatter(
        optimiser["resource_index"],
        optimiser["mean_daily_loss_mwh"],
        c=optimiser["cap_total_mvar"],
        cmap=cmap,
        norm=norm,
        marker="o",
        s=42,
        edgecolor=dark,
        linewidth=0.55,
        label="Optimiser candidate",
        zorder=3,
    )
    ax_global.scatter(
        path["resource_index"],
        path["mean_daily_loss_mwh"],
        c=path["cap_total_mvar"],
        cmap=cmap,
        norm=norm,
        marker="D",
        s=38,
        edgecolor="white",
        linewidth=0.55,
        zorder=2,
    )
    ax_global.scatter(
        [minimum["resource_index"]],
        [minimum["mean_daily_loss_mwh"]],
        marker="D",
        s=90,
        color=GREEN,
        edgecolor=dark,
        linewidth=0.9,
        label="Selected minimum",
        zorder=6,
    )
    ax_global.scatter(
        [pareto["resource_index"]],
        [pareto["mean_daily_loss_mwh"]],
        marker="D",
        s=80,
        color=pareto_colour,
        edgecolor=dark,
        linewidth=0.9,
        label="Pareto alternative",
        zorder=6,
    )
    ax_global.scatter(
        [high["resource_index"]],
        [high["mean_daily_loss_mwh"]],
        marker="D",
        s=86,
        color=GREY,
        edgecolor=dark,
        linewidth=0.9,
        zorder=6,
    )
    ax_global.text(
        float(high["resource_index"]) - 2.0,
        float(high["mean_daily_loss_mwh"]) + 0.105,
        "High-redundancy\nendpoint",
        ha="right",
        va="top",
        fontsize=6.7,
        color=GREY,
    )
    ax_global.text(
        0.025,
        0.965,
        "37 records (36 unique configurations)\n"
        "200 distinct profiles per seed; 0 observed event-days",
        transform=ax_global.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color=dark,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#909090", "linewidth": 0.7},
    )
    ax_global.set_xlim(67.0, 156.0)
    ax_global.set_ylim(0.82, 2.48)
    ax_global.set_xlabel("Normalised resource index")
    ax_global.set_ylabel(r"Mean daily line-loss energy (MWh day$^{-1}$)")
    ax_global.set_title("(a) Confirmed resource--loss landscape", loc="left", pad=8)
    ax_global.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.76),
        ncol=2,
        fontsize=6.6,
        markerscale=0.72,
        handletextpad=0.45,
        columnspacing=0.9,
        labelspacing=0.42,
        facecolor="white",
        edgecolor="#A0A0A0",
        borderpad=0.36,
    )
    ax_global.set_axisbelow(True)
    ax_global.grid(axis="y", color="#BEBEBE")

    # (b) Dedicated low-resource panel replaces the crowded inset.
    low_optimiser = optimiser[optimiser["resource_index"] <= 75.0]
    low_path = path[path["resource_index"] <= 75.2]
    if len(low_path) > 1:
        ax_low.plot(
            low_path["resource_index"],
            low_path["mean_daily_loss_mwh"],
            color="#7B8491",
            linewidth=1.1,
            linestyle="--",
            alpha=0.90,
            zorder=1,
        )
    ax_low.scatter(
        low_optimiser["resource_index"],
        low_optimiser["mean_daily_loss_mwh"],
        c=low_optimiser["cap_total_mvar"],
        cmap=cmap,
        norm=norm,
        marker="o",
        s=40,
        edgecolor=dark,
        linewidth=0.55,
        zorder=3,
    )
    ax_low.scatter(
        low_path["resource_index"],
        low_path["mean_daily_loss_mwh"],
        c=low_path["cap_total_mvar"],
        cmap=cmap,
        norm=norm,
        marker="D",
        s=38,
        edgecolor="white",
        linewidth=0.55,
        zorder=2,
    )
    ax_low.scatter(
        [minimum["resource_index"]],
        [minimum["mean_daily_loss_mwh"]],
        marker="D",
        s=105,
        color=GREEN,
        edgecolor=dark,
        linewidth=0.9,
        zorder=6,
    )
    ax_low.scatter(
        [pareto["resource_index"]],
        [pareto["mean_daily_loss_mwh"]],
        marker="D",
        s=95,
        color=pareto_colour,
        edgecolor=dark,
        linewidth=0.9,
        zorder=6,
    )
    ax_low.add_patch(
        FancyArrowPatch(
            (float(minimum["resource_index"]), float(minimum["mean_daily_loss_mwh"])),
            (float(pareto["resource_index"]), float(pareto["mean_daily_loss_mwh"])),
            arrowstyle="-|>",
            mutation_scale=8,
            linestyle="--",
            linewidth=0.9,
            color=dark,
            zorder=5,
        )
    )
    ax_low.text(
        70.8,
        0.985,
        f"Pareto alternative: +0.5 resource index\n{pareto_loss_change:.2f}% mean line loss",
        ha="left",
        va="center",
        fontsize=7.3,
        color=dark,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#808080", "linewidth": 0.7},
    )
    ax_low.set_xlim(69.6, 75.2)
    ax_low.set_ylim(0.84, 1.055)
    ax_low.set_xlabel("Normalised resource index")
    ax_low.set_ylabel(r"Mean daily loss (MWh day$^{-1}$)")
    ax_low.set_title("(b) Low-resource decision region", loc="left", pad=8)
    ax_low.set_axisbelow(True)
    ax_low.grid(color="#BEBEBE")

    colour_bar = fig.colorbar(optimiser_scatter, cax=ax_colourbar)
    colour_bar.set_label("Capacitor capacity (MVAr)", fontweight="bold")
    colour_bar.set_ticks((0.0, 0.25, 0.5, 0.75, 1.0))
    colour_bar.ax.tick_params(labelsize=8.0)
    colour_bar.outline.set_linewidth(0.8)

    fig.subplots_adjust(left=0.078, right=0.905, bottom=0.18, top=0.89)
    export_figure_fixed_size(fig, "fig_resource_efficiency")


def composition_aware_resource_efficiency() -> None:
    """Show why resource composition matters beyond the scalar resource index."""
    refine = pd.read_csv(PLANNING / "refine" / "wg_projected_candidate_summary.csv")
    confirm = pd.read_csv(PLANNING / "confirm" / "wg_projected_candidate_summary.csv")

    cap_zero = refine[np.isclose(refine["cap_total_mvar"], 0.0)].copy()
    cap_quarter = refine[np.isclose(refine["cap_total_mvar"], 0.25)].copy()
    comparison = cap_zero.merge(
        cap_quarter,
        on=["pv_s_scale", "svc_q_scale"],
        suffixes=("_cap0", "_cap025"),
    )
    comparison["delta_loss_mwh"] = (
        comparison["mean_daily_loss_mwh_cap025"] - comparison["mean_daily_loss_mwh_cap0"]
    )
    comparison["delta_intervention"] = (
        comparison["mean_intervention_rate_cap025"] - comparison["mean_intervention_rate_cap0"]
    )

    optimiser = confirm[confirm["selection_role"] == "optimiser_candidate"].copy()
    optimiser["resource_key"] = optimiser["resource_index"].round(6)
    pair_rows: list[dict[str, float]] = []
    for resource, subset in optimiser.groupby("resource_key"):
        if len(subset) < 2:
            continue
        best = subset.loc[subset["mean_daily_loss_mwh"].idxmin()]
        comparator = subset.loc[subset["mean_daily_loss_mwh"].idxmax()]
        saving = 100.0 * (
            float(comparator["mean_daily_loss_mwh"]) - float(best["mean_daily_loss_mwh"])
        ) / float(comparator["mean_daily_loss_mwh"])
        pair_rows.append(
            {
                "resource_index": float(resource),
                "best_loss_mwh": float(best["mean_daily_loss_mwh"]),
                "comparator_loss_mwh": float(comparator["mean_daily_loss_mwh"]),
                "saving_pct": saving,
                "best_pv_scale": float(best["pv_s_scale"]),
                "best_svc_scale": float(best["svc_q_scale"]),
                "best_cap_mvar": float(best["cap_total_mvar"]),
                "comparator_pv_scale": float(comparator["pv_s_scale"]),
                "comparator_svc_scale": float(comparator["svc_q_scale"]),
                "comparator_cap_mvar": float(comparator["cap_total_mvar"]),
            }
        )
    pairs = pd.DataFrame(pair_rows).sort_values("resource_index")

    source_a = cap_zero[
        ["candidate_id", "pv_s_scale", "svc_q_scale", "cap_total_mvar", "mean_daily_loss_mwh", "resource_index"]
    ].copy()
    source_a.insert(0, "panel", "a_refine_cap0_loss")
    source_b = comparison[
        ["pv_s_scale", "svc_q_scale", "delta_loss_mwh", "delta_intervention"]
    ].copy()
    source_b.insert(0, "panel", "b_refine_cap_increment")
    source_c = pairs.copy()
    source_c.insert(0, "panel", "c_confirm_equal_resource_pairs")
    pd.concat([source_a, source_b, source_c], ignore_index=True, sort=False).to_csv(
        OUT / "fig_composition_aware_efficiency_source.csv", index=False
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.titlesize": 8.6,
            "axes.labelsize": 8.0,
            "axes.labelweight": "bold",
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.9,
            "legend.fontsize": 6.6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(8.5, 3.05),
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.16)},
    )
    ax_loss, ax_delta, ax_pairs = axes
    dark = "#303030"

    pv_values = np.sort(cap_zero["pv_s_scale"].unique())
    svc_values = np.sort(cap_zero["svc_q_scale"].unique())
    loss_grid = cap_zero.pivot(index="svc_q_scale", columns="pv_s_scale", values="mean_daily_loss_mwh").reindex(
        index=svc_values, columns=pv_values
    )
    image_loss = ax_loss.pcolormesh(
        pv_values,
        svc_values,
        loss_grid.to_numpy(),
        shading="nearest",
        cmap="viridis",
        edgecolors=(1, 1, 1, 0.18),
        linewidth=0.25,
    )
    pv_mesh, svc_mesh = np.meshgrid(pv_values, svc_values)
    resource_mesh = 50.0 * (pv_mesh + svc_mesh)
    masked_resource = np.ma.masked_where(np.isnan(loss_grid.to_numpy()), resource_mesh)
    contours = ax_loss.contour(
        pv_mesh,
        svc_mesh,
        masked_resource,
        levels=(75, 80, 85, 90),
        colors="white",
        linewidths=0.65,
        linestyles="--",
        alpha=0.92,
    )
    ax_loss.clabel(contours, fmt=lambda value: f"J={value:.0f}", fontsize=5.5, inline=True)
    ax_loss.scatter(
        [0.7], [0.7], marker="D", s=46, color=GREEN, edgecolor=dark, linewidth=0.7, zorder=5
    )
    cax_loss = ax_loss.inset_axes([0.58, 0.67, 0.34, 0.035])
    colour_loss = fig.colorbar(image_loss, cax=cax_loss, orientation="horizontal")
    cax_loss.set_title("Loss (MWh/day)", fontsize=5.8, pad=3)
    colour_loss.ax.tick_params(labelsize=5.8, length=2, pad=1.5)
    ax_loss.set_xlabel("PV inverter capacity scale")
    ax_loss.set_ylabel("SVC capacity scale")
    ax_loss.set_title("(a) Low-resource loss landscape", loc="left", pad=6)
    ax_loss.text(
        0.03,
        0.96,
        "30 days, 3 seeds\ncap = 0 MVAr",
        transform=ax_loss.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=6.2,
        bbox={"boxstyle": "round,pad=0.20", "facecolor": (0, 0, 0, 0.42), "edgecolor": "none"},
    )

    delta_grid = comparison.pivot(index="svc_q_scale", columns="pv_s_scale", values="delta_loss_mwh").reindex(
        index=svc_values, columns=pv_values
    )
    delta_values = delta_grid.to_numpy()
    delta_cmap = LinearSegmentedColormap.from_list("loss_delta", (BLUE, "#F7F7F7", ORANGE))
    delta_norm = TwoSlopeNorm(
        vmin=float(np.nanmin(delta_values)),
        vcenter=0.0,
        vmax=float(np.nanmax(delta_values)),
    )
    image_delta = ax_delta.pcolormesh(
        pv_values,
        svc_values,
        delta_values,
        shading="nearest",
        cmap=delta_cmap,
        norm=delta_norm,
        edgecolors=(1, 1, 1, 0.25),
        linewidth=0.28,
    )
    beneficial = comparison[comparison["delta_loss_mwh"] < 0]
    ax_delta.scatter(
        beneficial["pv_s_scale"],
        beneficial["svc_q_scale"],
        s=32,
        marker="o",
        facecolor="none",
        edgecolor=GREEN,
        linewidth=1.0,
        zorder=5,
    )
    cax_delta = ax_delta.inset_axes([0.58, 0.67, 0.34, 0.035])
    colour_delta = fig.colorbar(image_delta, cax=cax_delta, orientation="horizontal")
    cax_delta.set_title("$\Delta$ loss (MWh/day)", fontsize=5.8, pad=3)
    colour_delta.set_ticks((float(np.nanmin(delta_values)), 0.0, float(np.nanmax(delta_values))))
    colour_delta.set_ticklabels(
        (f"{float(np.nanmin(delta_values)):.3f}", "0", f"{float(np.nanmax(delta_values)):.3f}")
    )
    colour_delta.ax.tick_params(labelsize=5.8, length=2, pad=1.5)
    ax_delta.set_xlabel("PV inverter capacity scale")
    ax_delta.set_ylabel("SVC capacity scale")
    ax_delta.set_title("(b) Value of the first 0.25 MVAr", loc="left", pad=6)
    ax_delta.text(
        0.04,
        0.96,
        f"{len(beneficial)}/{len(comparison)} configurations\nlower line loss",
        transform=ax_delta.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=dark,
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#909090", "linewidth": 0.6},
    )

    row_y = np.arange(len(pairs))[::-1]
    ax_pairs.hlines(
        row_y,
        pairs["best_loss_mwh"],
        pairs["comparator_loss_mwh"],
        color="#9CA3AF",
        linewidth=1.2,
        zorder=1,
    )
    ax_pairs.scatter(
        pairs["best_loss_mwh"],
        row_y,
        s=34,
        marker="o",
        color=BLUE,
        edgecolor="white",
        linewidth=0.55,
        label="PV-heavier allocation",
        zorder=3,
    )
    ax_pairs.scatter(
        pairs["comparator_loss_mwh"],
        row_y,
        s=34,
        marker="s",
        color=ORANGE,
        edgecolor="white",
        linewidth=0.55,
        label="Comparator at same $J_{res}$",
        zorder=3,
    )
    for y_pos, row in zip(row_y, pairs.itertuples(index=False)):
        ax_pairs.text(
            float(row.comparator_loss_mwh) + 0.0025,
            y_pos,
            f"{float(row.saving_pct):.2f}%",
            va="center",
            ha="left",
            fontsize=6.2,
            fontweight="bold",
            color=BLUE,
        )
    ax_pairs.set_yticks(row_y, [f"J={value:.2f}" for value in pairs["resource_index"]])
    x_min = float(pairs["best_loss_mwh"].min()) - 0.005
    x_max = float(pairs["comparator_loss_mwh"].max()) + 0.035
    ax_pairs.set_xlim(x_min, x_max)
    ax_pairs.set_xlabel(r"Confirmed line loss (MWh day$^{-1}$)")
    ax_pairs.set_title("(c) Equal-resource allocation pairs", loc="left", pad=6)
    ax_pairs.legend(
        loc="lower right",
        facecolor="white",
        edgecolor="#A0A0A0",
        borderpad=0.35,
        handletextpad=0.4,
    )
    ax_pairs.text(
        0.98,
        0.97,
        "200 distinct profiles, 3 seeds",
        transform=ax_pairs.transAxes,
        ha="right",
        va="top",
        fontsize=6.3,
        color=GREY,
    )

    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(False)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
    ax_pairs.grid(axis="x", color="#C7CBD1", linestyle="--", linewidth=0.55, alpha=0.55)
    fig.subplots_adjust(left=0.068, right=0.988, bottom=0.18, top=0.88, wspace=0.34)
    export_figure_fixed_size(fig, "fig_composition_aware_efficiency")


def three_stage_screening_fidelity() -> None:
    """Audit whether early-stage estimates preserve confirmed candidate ordering."""
    screen = pd.read_csv(PLANNING / "screen" / "wg_projected_candidate_summary.csv")
    refine = pd.read_csv(PLANNING / "refine" / "wg_projected_candidate_summary.csv")
    confirm = pd.read_csv(PLANNING / "confirm" / "wg_projected_candidate_summary.csv")
    confirm = confirm[confirm["selection_role"] == "optimiser_candidate"].copy()

    columns = ["candidate_id", "pv_s_scale", "svc_q_scale", "cap_total_mvar", "resource_index", "mean_daily_loss_mwh", "mean_intervention_rate"]
    merged = confirm[columns].rename(
        columns={
            "mean_daily_loss_mwh": "confirm_loss_mwh",
            "mean_intervention_rate": "confirm_intervention_rate",
        }
    )
    for stage_name, stage_data in (("screen", screen), ("refine", refine)):
        stage_columns = stage_data[["candidate_id", "mean_daily_loss_mwh", "mean_intervention_rate"]].rename(
            columns={
                "mean_daily_loss_mwh": f"{stage_name}_loss_mwh",
                "mean_intervention_rate": f"{stage_name}_intervention_rate",
            }
        )
        merged = merged.merge(stage_columns, on="candidate_id", how="inner")
    merged = merged.sort_values("confirm_loss_mwh").reset_index(drop=True)
    merged.to_csv(OUT / "fig_three_stage_fidelity_source.csv", index=False)

    loss_rho_screen = merged["screen_loss_mwh"].corr(merged["confirm_loss_mwh"], method="spearman")
    loss_rho_refine = merged["refine_loss_mwh"].corr(merged["confirm_loss_mwh"], method="spearman")
    effort_rho_screen = merged["screen_intervention_rate"].corr(
        merged["confirm_intervention_rate"], method="spearman"
    )
    effort_rho_refine = merged["refine_intervention_rate"].corr(
        merged["confirm_intervention_rate"], method="spearman"
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.titlesize": 8.6,
            "axes.labelsize": 8.0,
            "axes.labelweight": "bold",
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.9,
            "legend.fontsize": 6.6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(8.5, 3.0),
        gridspec_kw={"width_ratios": (1.16, 1.0, 1.0)},
    )
    ax_track, ax_loss, ax_effort = axes
    stage_x = np.arange(3)
    stage_labels = ("Screen\n3 days", "Refine\n30 days", "Confirm\n200 days")
    for row in merged.itertuples(index=False):
        values = [row.screen_loss_mwh, row.refine_loss_mwh, row.confirm_loss_mwh]
        is_minimum = int(row.candidate_id) == 0
        is_pareto = int(row.candidate_id) == 400
        colour = GREEN if is_minimum else BLUE if is_pareto else "#AEB4BC"
        width = 1.65 if (is_minimum or is_pareto) else 0.65
        alpha = 1.0 if (is_minimum or is_pareto) else 0.62
        zorder = 4 if (is_minimum or is_pareto) else 1
        ax_track.plot(
            stage_x,
            values,
            marker="o",
            markersize=3.2 if (is_minimum or is_pareto) else 2.2,
            color=colour,
            linewidth=width,
            alpha=alpha,
            zorder=zorder,
        )
        if is_minimum or is_pareto:
            ax_track.annotate(
                "Selected minimum" if is_minimum else "Pareto alternative",
                xy=(2.0, row.confirm_loss_mwh),
                xytext=(6, 6 if is_minimum else -8),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=6.0,
                color=colour,
            )
    ax_track.set_xticks(stage_x, stage_labels)
    ax_track.set_xlim(-0.12, 2.72)
    ax_track.set_ylabel(r"Mean daily line loss (MWh day$^{-1}$)")
    ax_track.set_title("(a) Candidate estimates across stages", loc="left", pad=6)
    ax_track.text(
        0.03,
        0.97,
        "20 retained candidates\n"
        + rf"$\rho_{{S,C}}$={loss_rho_screen:.3f}; $\rho_{{R,C}}$={loss_rho_refine:.3f}",
        transform=ax_track.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#303030",
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#909090", "linewidth": 0.6},
    )

    def agreement_panel(
        ax: plt.Axes,
        screen_values: pd.Series,
        refine_values: pd.Series,
        confirm_values: pd.Series,
        title: str,
        label: str,
        rho_screen: float,
        rho_refine: float,
        scale: float = 1.0,
        bias_screen: float | None = None,
        bias_refine: float | None = None,
    ) -> None:
        screen_plot = screen_values.to_numpy() * scale
        refine_plot = refine_values.to_numpy() * scale
        confirm_plot = confirm_values.to_numpy() * scale
        all_values = np.concatenate((screen_plot, refine_plot, confirm_plot))
        value_min = float(all_values.min())
        value_max = float(all_values.max())
        span = max(value_max - value_min, 1e-6)
        lower = max(0.0, value_min - 0.08 * span) if value_min >= 0 else value_min - 0.08 * span
        upper = value_max + 0.10 * span
        ax.plot((lower, upper), (lower, upper), color="#4B5563", linestyle="--", linewidth=0.9, zorder=1)
        ax.scatter(
            screen_plot,
            confirm_plot,
            s=28,
            marker="o",
            facecolor="white",
            edgecolor=ORANGE,
            linewidth=0.9,
            label="Screen vs confirm",
            zorder=3,
        )
        ax.scatter(
            refine_plot,
            confirm_plot,
            s=28,
            marker="^",
            facecolor=BLUE,
            edgecolor="white",
            linewidth=0.45,
            label="Refine vs confirm",
            zorder=3,
        )
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_xlabel(f"Early-stage {label}")
        ax.set_ylabel(f"Confirmed {label}")
        ax.set_title(title, loc="left", pad=6)
        annotation = f"Spearman $\\rho$\nScreen: {rho_screen:.3f}\nRefine: {rho_refine:.3f}"
        if bias_screen is not None and bias_refine is not None:
            annotation = (
                f"Screen: $\\rho$={rho_screen:.3f}, bias {bias_screen:+.3f} pp\n"
                f"Refine: $\\rho$={rho_refine:.3f}, bias {bias_refine:+.3f} pp"
            )
        ax.text(
            0.04,
            0.96,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.1,
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#909090", "linewidth": 0.6},
        )
        ax.grid(True, color="#C7CBD1", linestyle="--", linewidth=0.55, alpha=0.55)
        ax.set_aspect("equal", adjustable="box")

    agreement_panel(
        ax_loss,
        merged["screen_loss_mwh"],
        merged["refine_loss_mwh"],
        merged["confirm_loss_mwh"],
        "(b) Line-loss agreement",
        "loss (MWh/day)",
        loss_rho_screen,
        loss_rho_refine,
    )
    agreement_panel(
        ax_effort,
        merged["screen_intervention_rate"],
        merged["refine_intervention_rate"],
        merged["confirm_intervention_rate"],
        "(c) Projection-use ranking and bias",
        "intervention (%)",
        effort_rho_screen,
        effort_rho_refine,
        scale=100.0,
        bias_screen=100.0
        * (merged["screen_intervention_rate"] - merged["confirm_intervention_rate"]).mean(),
        bias_refine=100.0
        * (merged["refine_intervention_rate"] - merged["confirm_intervention_rate"]).mean(),
    )
    ax_loss.legend(
        loc="lower right",
        facecolor="white",
        edgecolor="#A0A0A0",
        borderpad=0.35,
        handletextpad=0.4,
    )
    for ax in axes:
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
    ax_track.grid(axis="y", color="#C7CBD1", linestyle="--", linewidth=0.55, alpha=0.55)
    fig.subplots_adjust(left=0.07, right=0.988, bottom=0.19, top=0.88, wspace=0.37)
    export_figure_fixed_size(fig, "fig_three_stage_fidelity")


def run_trace(projected: bool, day: int, seed: int) -> pd.DataFrame:
    theta = np.asarray([0.7, 0.7, 0.0], dtype=np.float32)
    env = build_env(33, theta, [20, 8], 1.0)
    run_dir = ROOT / "models" / "wg_cvar" / f"seed{seed}"
    device = torch.device("cpu")
    actor, _ = load_actor(run_dir, len(env.observation_space), len(env.action_space), device)
    projector = ACPowerFlowSafetyProjector()
    obs = env.reset_at_step(day * STEPS_PER_DAY)
    rows = []
    theta_tensor = torch.as_tensor(theta, dtype=torch.float32)
    for step in range(STEPS_PER_DAY):
        with torch.no_grad():
            raw_action, _ = actor._pi(torch.as_tensor(obs, dtype=torch.float32), theta_tensor)
        policy_action = raw_action.numpy()
        if projected:
            result = projector.project(env, policy_action)
            executed_action = result.action
            intervened = result.intervened
            correction_norm = result.correction_norm
        else:
            executed_action = policy_action
            intervened = False
            correction_norm = 0.0
        next_obs, _, _, _, _, _, vmax, vmin, grid_loss, new_state = env.step_model(executed_action)
        n_bus = len(env.model.bus)
        q_mvar = np.asarray(next_obs[-len(env.action_space) :], dtype=float)
        raw_q_mvar = env.action_clip(policy_action)
        row = {
            "controller": "Projected WG-CVaR" if projected else "Raw WG-CVaR",
            "seed": seed,
            "day_index": day,
            "step": step,
            "time_hour": step * DT_HOURS,
            "vmin_pu": float(vmin),
            "vmax_pu": float(vmax),
            "critical_bus": int(np.argmin(np.asarray(next_obs[:n_bus], dtype=float))) + 1,
            "line_loss_mw": max(0.0, float(-grid_loss)),
            "intervened": int(intervened),
            "correction_norm": float(correction_norm),
        }
        for index, label in enumerate(DEVICE_LABELS):
            key = label.lower().replace("-", "_")
            row[f"{key}_policy_q_mvar"] = float(raw_q_mvar[index])
            row[f"{key}_executed_q_mvar"] = float(q_mvar[index])
            row[f"{key}_correction_q_mvar"] = float(q_mvar[index] - raw_q_mvar[index])
        rows.append(row)
        obs = new_state
    return pd.DataFrame(rows)


def worst_day_trace() -> None:
    day = 273
    seed = 42
    source_path = OUT / "fig_worst_day_trace_source.csv"
    if source_path.exists():
        data = pd.read_csv(source_path)
    else:
        raw = run_trace(False, day, seed)
        projected = run_trace(True, day, seed)
        data = pd.concat([raw, projected], ignore_index=True)
        data.to_csv(source_path, index=False)

    raw_data = data[data["controller"] == "Raw WG-CVaR"]
    proj_data = data[data["controller"] == "Projected WG-CVaR"]
    time = raw_data["time_hour"].to_numpy(dtype=float)
    raw_voltage = raw_data["vmin_pu"].to_numpy(dtype=float)
    projected_voltage = proj_data["vmin_pu"].to_numpy(dtype=float)
    raw_loss = raw_data["line_loss_mw"].to_numpy(dtype=float)
    projected_loss = proj_data["line_loss_mw"].to_numpy(dtype=float)
    intervention = proj_data["intervened"].astype(bool).to_numpy()
    intervention_indices = np.flatnonzero(intervention)
    if len(intervention_indices) != 1:
        raise RuntimeError(f"Expected one intervention step, found {len(intervention_indices)}")
    event_index = int(intervention_indices[0])
    event_time = float(time[event_index])

    raw_vmin = float(np.min(raw_voltage))
    projected_vmin = float(np.min(projected_voltage))
    voltage_recovery = projected_vmin - raw_vmin
    raw_daily_loss = float(np.sum(raw_loss) * DT_HOURS)
    projected_daily_loss = float(np.sum(projected_loss) * DT_HOURS)
    daily_loss_delta = projected_daily_loss - raw_daily_loss
    daily_loss_delta_pct = 100.0 * daily_loss_delta / raw_daily_loss

    correction_values = []
    for label in DEVICE_LABELS:
        key = label.lower().replace("-", "_")
        correction_values.append(float(proj_data.iloc[event_index][f"{key}_correction_q_mvar"]))
    correction_values = np.asarray(correction_values, dtype=float)
    dominant_share = 100.0 * abs(correction_values[0]) / np.sum(np.abs(correction_values))

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 1.1,
            "legend.fontsize": 8.0,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
        }
    )

    dark = "#303030"
    raw_colour = "#6B7280"
    limit_colour = "#B94A48"
    fig = plt.figure(figsize=(8.5, 5.15))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.18, 1.0), hspace=0.42, wspace=0.30)
    ax_voltage = fig.add_subplot(grid[0, :])
    ax_devices = fig.add_subplot(grid[1, 0])
    ax_loss_delta = fig.add_subplot(grid[1, 1])

    # (a) Full-day voltage trajectory, with a focused intervention inset.
    ax_voltage.axhspan(0.94, 0.95, color=limit_colour, alpha=0.055, zorder=0)
    ax_voltage.plot(
        time,
        raw_voltage,
        linestyle="--",
        color=raw_colour,
        linewidth=1.55,
        label="Raw WG-CVaR",
        zorder=2,
    )
    ax_voltage.plot(
        time,
        projected_voltage,
        color=ORANGE,
        linewidth=1.8,
        label="WG-CVaR + projection",
        zorder=3,
    )
    ax_voltage.axhline(
        0.95,
        color=limit_colour,
        linestyle=(0, (4, 3)),
        linewidth=1.15,
        label="Reported voltage limit",
        zorder=1,
    )
    ax_voltage.axvspan(
        event_time - DT_HOURS / 2,
        event_time + DT_HOURS / 2,
        color=ORANGE,
        alpha=0.12,
        linewidth=0,
        zorder=1,
    )
    ax_voltage.scatter(
        [event_time, event_time],
        [raw_voltage[event_index], projected_voltage[event_index]],
        s=26,
        facecolors=("white", ORANGE),
        edgecolors=(raw_colour, dark),
        linewidths=0.9,
        zorder=5,
    )
    ax_voltage.annotate(
        "Projection active at 10:30",
        xy=(event_time, projected_voltage[event_index]),
        xytext=(12.3, 0.9575),
        ha="left",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=dark,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#8A8A8A", "linewidth": 0.8},
        arrowprops={"arrowstyle": "->", "color": dark, "linewidth": 0.9},
        zorder=6,
    )
    ax_voltage.set_xlim(0.0, 23.75)
    ax_voltage.set_ylim(0.940, 1.003)
    ax_voltage.set_xticks(np.arange(0, 25, 4))
    ax_voltage.set_ylabel("Minimum bus voltage (pu)")
    ax_voltage.set_title("(a) Voltage restoration on the worst confirmed day", loc="left", pad=8)
    ax_voltage.text(
        0.01,
        0.96,
        "Seed 42; held-out day 273",
        transform=ax_voltage.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color=raw_colour,
    )
    ax_voltage.legend(
        loc="upper right",
        ncol=3,
        facecolor="white",
        edgecolor="#A0A0A0",
        borderpad=0.45,
        columnspacing=1.0,
    )
    ax_voltage.set_axisbelow(True)
    ax_voltage.grid(axis="y", color="#BEBEBE")

    # Keep the inset clear of the parent y-axis labels and tick marks.
    zoom = ax_voltage.inset_axes([0.060, 0.140, 0.31, 0.48])
    zoom_mask = (time >= 9.75) & (time <= 11.25)
    zoom.plot(time[zoom_mask], raw_voltage[zoom_mask], "--", color=raw_colour, linewidth=1.25)
    zoom.plot(time[zoom_mask], projected_voltage[zoom_mask], color=ORANGE, linewidth=1.45)
    zoom.axhline(0.95, color=limit_colour, linestyle=(0, (4, 3)), linewidth=0.9)
    zoom.scatter(
        [event_time, event_time],
        [raw_voltage[event_index], projected_voltage[event_index]],
        s=20,
        facecolors=("white", ORANGE),
        edgecolors=(raw_colour, dark),
        linewidths=0.8,
        zorder=4,
    )
    zoom.annotate(
        "",
        xy=(event_time + 0.04, projected_voltage[event_index]),
        xytext=(event_time + 0.04, raw_voltage[event_index]),
        arrowprops={"arrowstyle": "<->", "color": dark, "linewidth": 0.8},
    )
    zoom.text(
        event_time + 0.12,
        0.9470,
        f"+{voltage_recovery:.5f} pu",
        fontsize=6.8,
        fontweight="bold",
        color=dark,
        ha="left",
        va="center",
    )
    zoom.set_xlim(9.75, 11.25)
    zoom.set_ylim(0.942, 0.9565)
    zoom.set_xticks((10.0, 10.5, 11.0))
    zoom.set_yticks((0.944, 0.950, 0.956))
    zoom.tick_params(labelsize=6.2, length=2.3, pad=1.5)
    zoom.set_title("Intervention detail", fontsize=7.2, pad=2.5)
    zoom.grid(axis="y", color="#D0D0D0", linewidth=0.45, alpha=0.6)

    # (b) Device-level allocation at the single corrected step.
    device_colours = (ORANGE, BLUE, GREEN, GREY)
    device_x = np.arange(len(DEVICE_LABELS), dtype=float)
    bars = ax_devices.bar(
        device_x,
        correction_values,
        width=0.58,
        color=device_colours,
        edgecolor=dark,
        linewidth=0.8,
        alpha=0.90,
        zorder=2,
    )
    for bar, value in zip(bars, correction_values):
        ax_devices.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color=dark,
        )
    ax_devices.text(
        0.98,
        0.95,
        f"IBVR-17 supplies {dominant_share:.1f}%\nof total correction",
        transform=ax_devices.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        color=dark,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#808080", "linewidth": 0.8},
    )
    ax_devices.set_xticks(device_x, DEVICE_LABELS)
    ax_devices.set_ylim(0.0, 0.59)
    ax_devices.set_ylabel(r"Reactive-power correction $\Delta Q$ (MVAr)")
    ax_devices.set_title("(b) Device allocation at 10:30", loc="left", pad=8)
    ax_devices.set_axisbelow(True)
    ax_devices.grid(axis="y", color="#BEBEBE")

    # (c) Incremental loss makes the physical cost of projection explicit.
    loss_delta_kw = 1000.0 * (projected_loss - raw_loss)
    ax_loss_delta.plot(time, loss_delta_kw, color=dark, linewidth=1.25, zorder=3)
    ax_loss_delta.fill_between(
        time,
        0.0,
        loss_delta_kw,
        where=loss_delta_kw >= 0.0,
        color=ORANGE,
        alpha=0.28,
        interpolate=True,
        zorder=2,
    )
    ax_loss_delta.fill_between(
        time,
        0.0,
        loss_delta_kw,
        where=loss_delta_kw < 0.0,
        color=BLUE,
        alpha=0.22,
        interpolate=True,
        zorder=2,
    )
    ax_loss_delta.axhline(0.0, color="#8A8A8A", linewidth=0.9)
    ax_loss_delta.axvspan(
        event_time - DT_HOURS / 2,
        event_time + DT_HOURS / 2,
        color=ORANGE,
        alpha=0.10,
        linewidth=0,
        zorder=1,
    )
    ax_loss_delta.text(
        0.98,
        0.95,
        f"Daily energy change\n{daily_loss_delta:+.5f} MWh ({daily_loss_delta_pct:+.3f}%)",
        transform=ax_loss_delta.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        color=dark,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#808080", "linewidth": 0.8},
    )
    ax_loss_delta.set_xlim(0.0, 23.75)
    ax_loss_delta.set_xticks(np.arange(0, 25, 4))
    ax_loss_delta.set_xlabel("Time of day (h)")
    ax_loss_delta.set_ylabel("Projected minus raw loss (kW)")
    ax_loss_delta.set_title("(c) Incremental line-loss effect", loc="left", pad=8)
    ax_loss_delta.set_axisbelow(True)
    ax_loss_delta.grid(axis="y", color="#BEBEBE")
    ax_devices.set_xlabel("Reactive-power device")

    fig.subplots_adjust(left=0.075, right=0.988, bottom=0.105, top=0.93)
    export_figure_fixed_size(fig, "fig_worst_day_trace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-trace", action="store_true", help="Skip the AC replay figure")
    args = parser.parse_args()
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    framework_diagram()
    locked_generalisation()
    locked_generalisation_nature()
    projection_ablation()
    projection_ablation_nature()
    projection_ablation_distribution()
    capacity_planning()
    resource_efficiency()
    composition_aware_resource_efficiency()
    if not args.skip_trace:
        worst_day_trace()
    print(f"Figures and source data written to {OUT}")


if __name__ == "__main__":
    main()
