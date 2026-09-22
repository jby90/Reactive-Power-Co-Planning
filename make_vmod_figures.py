"""Create archival figures for the redesigned VMOD study using frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from vmod_current_study import (
    DEVELOPMENT_MARGIN_ROOT,
    DEVELOPMENT_PROTOCOL_33,
    DEVELOPMENT_PROTOCOL_69,
    EXTERNAL_PROTOCOL_33,
    FINAL_ROOT,
    PATH_MARGIN_SUMMARY,
    env69_run_dir,
    main_run_dir,
    read_selected_margin,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL = EXTERNAL_PROTOCOL_33
MARGIN_ROOT = DEVELOPMENT_MARGIN_ROOT
OUT = Path(
    os.environ.get(
        "VMOD_FIGURE_OUT",
        str(
            ROOT
            / "outputs"
            / "."
            / "."
            / "figures"
        ),
    )
)

ORANGE = "#E67E22"
BLUE = "#2980B9"
TEAL = "#1F8A83"
GREEN = "#2E8B57"
CHARCOAL = "#3F454D"
GREY = "#6B7280"
LIGHT_GREY = "#D1D5DB"
RED = "#C0392B"
SEED_COLOURS = ["#4E79A7", "#59A14F", "#E15759", "#B07AA1", "#76B7B2"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.0,
            "axes.labelweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(fig: plt.Figure, stem: str, inputs: list[Path]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 400}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        fig.savefig(
            OUT / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.04,
            **kwargs,
        )
    manifest = {
        "figure": stem,
        "backend": "Python/matplotlib",
        "inputs": {str(path): sha256(path) for path in inputs},
        "outputs": [f"{stem}.{suffix}" for suffix in ("pdf", "svg", "png", "tiff")],
    }
    (OUT / f"{stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    plt.close(fig)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required figure evidence is missing: {path}")
    return path


def read_json(path: Path) -> dict:
    return json.loads(require(path).read_text(encoding="utf-8"))


def all_capacity_feasible_days(
    episodes: pd.DataFrame, capacity_levels: int
) -> np.ndarray:
    """Return days with complete, event-free AC-OPF support at every capacity."""
    required = {"day", "candidate_id", "pf_fail", "risk"}
    missing = required.difference(episodes.columns)
    if missing:
        raise ValueError(f"AC-OPF episodes are missing columns: {sorted(missing)}")
    grouped = episodes.groupby("day").agg(
        records=("candidate_id", "size"),
        observed_capacity_levels=("candidate_id", "nunique"),
        power_flow_failures=("pf_fail", "sum"),
        voltage_events=("risk", "sum"),
    )
    return grouped.index[
        (grouped.records == int(capacity_levels))
        & (grouped.observed_capacity_levels == int(capacity_levels))
        & (grouped.power_flow_failures == 0)
        & (grouped.voltage_events == 0)
    ].to_numpy()


def selected_runs() -> tuple[float, Path, Path, dict]:
    margin = read_selected_margin()
    main_run = main_run_dir(margin)
    env69_run = env69_run_dir(margin)
    claim = read_json(FINAL_ROOT / "claim_gate_summary.json")
    return float(margin), main_run, env69_run, claim


def panel_label(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
    )
    ax.set_title(title, loc="left", pad=8)


def tidy_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=LIGHT_GREY, linestyle="--", linewidth=0.65, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def margin_label(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
    title_size: float = 8.8,
    body_size: float = 7.5,
) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.15,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.08),
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.76,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=CHARCOAL,
    )
    ax.text(
        x + width / 2,
        y + height * 0.35,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=CHARCOAL,
        linespacing=1.22,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.025),
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            color=GREY,
            shrinkA=2,
            shrinkB=2,
        )
    )
    if label:
        ax.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=7.1,
            color=GREY,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": LIGHT_GREY,
                "linewidth": 0.6,
            },
        )


def protocol_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.20, 4.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.965, "a", fontsize=10.5, fontweight="bold", va="top")
    ax.text(
        0.055,
        0.965,
        "Offline capacity-conditioned distillation",
        fontsize=10.2,
        fontweight="bold",
        va="top",
        color=CHARCOAL,
    )
    rounded_box(
        ax,
        (0.04, 0.59),
        0.20,
        0.25,
        "Physical inputs",
        "Nine nested ratings\n40 fitting days\nnonlinear feeder model",
        BLUE,
        title_size=8.0,
        body_size=7.1,
    )
    rounded_box(
        ax,
        (0.30, 0.59),
        0.22,
        0.25,
        "Margin AC-OPF teacher",
        "Loss-minimising dispatch\ninside 0.95+$\\delta$ to 1.05-$\\delta$\nfailed solves retained",
        CHARCOAL,
        title_size=7.7,
        body_size=6.9,
    )
    rounded_box(
        ax,
        (0.58, 0.59),
        0.20,
        0.25,
        "Five students",
        "State + capacity vector\n32 allocated train days\n8 allocated early-stop days",
        ORANGE,
        title_size=8.0,
        body_size=7.1,
    )
    rounded_box(
        ax,
        (0.84, 0.59),
        0.12,
        0.25,
        "Raw actor",
        "One model\nfor all tested\nratings",
        ORANGE,
        title_size=8.3,
        body_size=7.0,
    )
    arrow(ax, (0.24, 0.715), (0.30, 0.715))
    arrow(ax, (0.52, 0.715), (0.58, 0.715))
    arrow(ax, (0.78, 0.715), (0.84, 0.715))

    ax.plot([0.03, 0.97], [0.50, 0.50], color=LIGHT_GREY, linewidth=0.8)
    ax.text(0.02, 0.46, "b", fontsize=10.5, fontweight="bold", va="top")
    ax.text(
        0.055,
        0.46,
        "Disjoint evidence stages and online execution",
        fontsize=10.2,
        fontweight="bold",
        va="top",
        color=CHARCOAL,
    )
    rounded_box(
        ax,
        (0.035, 0.10),
        0.18,
        0.24,
        "Margin calibration",
        "Development window\n10 held-out days\nsmallest passing $\\delta$",
        BLUE,
        title_size=7.4,
        body_size=6.4,
    )
    rounded_box(
        ax,
        (0.265, 0.10),
        0.18,
        0.24,
        "Capacity selection",
        "External window\n17 held-out days\nfreeze adjacent pair",
        BLUE,
        title_size=7.4,
        body_size=6.4,
    )
    rounded_box(
        ax,
        (0.495, 0.10),
        0.18,
        0.24,
        "Confirmation",
        "External window\n349 held-out days\nseed-wise exact bounds",
        GREEN,
        title_size=7.7,
        body_size=6.4,
    )
    rounded_box(
        ax,
        (0.725, 0.10),
        0.24,
        0.24,
        "Online path",
        "Actor $\\rightarrow$ MVAr $\\rightarrow$ clipping\nnonlinear AC plant\nno OPF or safety projection",
        ORANGE,
        title_size=7.9,
        body_size=6.6,
    )
    arrow(ax, (0.215, 0.22), (0.265, 0.22), "freeze $\\delta$", (0.0, 0.145))
    arrow(ax, (0.445, 0.22), (0.495, 0.22), "freeze pair", (0.0, 0.145))
    arrow(ax, (0.675, 0.22), (0.725, 0.22), "evaluate", (0.0, 0.145))

    ax.text(
        0.5,
        0.015,
        "No margin-calibration, external-selection, or external-confirmation day is used to generate teacher data or fit actor parameters.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=CHARCOAL,
    )
    export(
        fig,
        "fig_vmod_protocol",
        [
            DEVELOPMENT_PROTOCOL_33 / "capacity_path.csv",
            DEVELOPMENT_PROTOCOL_33 / "student_training_days.csv",
            DEVELOPMENT_PROTOCOL_33 / "student_validation_days.csv",
            DEVELOPMENT_PROTOCOL_33 / "calibration_days.csv",
            PROTOCOL / "selection_days.csv",
            PROTOCOL / "confirmation_days.csv",
            DEVELOPMENT_PROTOCOL_33 / "protocol.json",
            PROTOCOL / "external_validation_protocol.json",
        ],
    )


def margin_and_teacher_figure() -> None:
    selected_margin = read_selected_margin()
    summary_path = PATH_MARGIN_SUMMARY
    margin_summary = read_json(summary_path)
    candidates = sorted(margin_summary["candidates"], key=lambda row: row["margin_pu"])
    event_matrix = []
    support_matrix = []
    input_paths = [summary_path]
    selected_candidate = None
    for candidate in candidates:
        event_matrix.append(
            [int(point["total_event_days"]) for point in candidate["points"]]
        )
        support_matrix.append(
            [bool(point["adequate_teacher_support"]) for point in candidate["points"]]
        )
        if np.isclose(float(candidate["margin_pu"]), selected_margin):
            selected_candidate = candidate
    if selected_candidate is None:
        raise RuntimeError("Selected margin is absent from calibration candidates")

    selected_root = Path(selected_candidate["candidate_dir"]).resolve()
    manifest_path = selected_root / "dataset" / "manifest.json"
    teacher_summary_path = selected_root / "teacher" / "summary.json"
    matched_path = selected_root / "teacher" / "capacity_summary_matched_days.csv"
    manifest = read_json(manifest_path)
    teacher_summary = read_json(teacher_summary_path)
    coverage = pd.DataFrame(manifest["candidate_coverage"]).sort_values("candidate_id")
    matched = pd.read_csv(require(matched_path)).sort_values("candidate_id")
    input_paths.extend([manifest_path, teacher_summary_path, matched_path])

    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.62), layout="constrained")

    ax = axes[0]
    event_array = np.asarray(event_matrix)
    event_max = int(event_array.max())
    event_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "event_days",
        ["#FFFFFF", "#FFF3BF", "#FDB863", "#E34A33", "#8E0152"],
    )
    image = ax.imshow(
        event_array,
        cmap=event_cmap,
        aspect="auto",
        vmin=0,
        vmax=event_max,
        interpolation="nearest",
    )
    path_indices = [int(point["path_index"]) for point in candidates[0]["points"]]
    ax.set_xticks(
        range(len(path_indices)),
        [f"P{index:02d}" for index in path_indices],
        rotation=45,
    )
    ax.set_yticks(
        range(len(candidates)),
        [margin_label(float(row["margin_pu"])) for row in candidates],
    )
    ax.set_xticks(np.arange(-0.5, len(path_indices), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(candidates), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("Capacity point")
    ax.set_ylabel("Margin $\\delta$ (pu)")
    panel_label(ax, "a", "Calibration events")
    for row_idx, row in enumerate(event_matrix):
        for col_idx, value in enumerate(row):
            marker = "" if support_matrix[row_idx][col_idx] else "$^{\\dagger}$"
            ax.text(
                col_idx,
                row_idx,
                f"{value}{marker}",
                ha="center",
                va="center",
                fontsize=6.8,
                color="white" if value >= 24 else CHARCOAL,
            )
    selected_row = next(
        i
        for i, row in enumerate(candidates)
        if np.isclose(float(row["margin_pu"]), selected_margin)
    )
    ax.add_patch(
        mpl.patches.Rectangle(
            (-0.49, selected_row - 0.49),
            len(path_indices) - 0.02,
            0.98,
            fill=False,
            edgecolor=CHARCOAL,
            linewidth=1.5,
        )
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_ticks(sorted({0, 10, 20, 30, event_max}))
    cbar.set_label("Event-days")
    ax.text(
        0.01,
        -0.34,
        "$^{\\dagger}$ teacher support below 32/40 days",
        transform=ax.transAxes,
        fontsize=6.4,
        color=GREY,
    )

    ax = axes[1]
    x = coverage["candidate_id"].to_numpy()
    y = coverage["converged_days"].to_numpy()
    colours = [GREEN if value >= 32 else LIGHT_GREY for value in y]
    ax.bar(x, y, color=colours, edgecolor=CHARCOAL, linewidth=0.6)
    ax.axhline(32, color=RED, linestyle="--", linewidth=1.0)
    ax.text(
        7.9,
        32.8,
        "32/40-day gate",
        ha="right",
        va="bottom",
        color=RED,
        fontsize=7.2,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": RED,
            "linewidth": 0.8,
            "alpha": 0.96,
        },
        zorder=5,
    )
    ax.set_xticks(x, [f"P{int(value):02d}" for value in x], rotation=45)
    ax.set_ylim(0, 42)
    ax.set_xlabel("Capacity point")
    ax.set_ylabel("Complete days / 40")
    panel_label(ax, "b", "Teacher support")
    tidy_axis(ax)

    ax = axes[2]
    x = matched["candidate_id"].to_numpy()
    y = matched["mean_daily_loss_mwh"].to_numpy()
    ax.plot(x, y, color=CHARCOAL, marker="D", markersize=4.2, linewidth=1.3)
    ax.set_xticks(x, [f"P{int(value):02d}" for value in x], rotation=45)
    ax.set_xlabel("Capacity point")
    ax.set_ylabel("AC-OPF loss (MWh/day)")
    panel_label(ax, "c", "Matched teacher loss")
    tidy_axis(ax)
    max_increase = float(teacher_summary["maximum_positive_loss_increase_mwh"])
    ax.text(
        0.98,
        0.72,
        f"Largest upward step\n{max_increase:.1e} MWh/day\n(within audit tolerance)",
        transform=ax.transAxes,
        fontsize=6.4,
        ha="right",
        va="top",
        color=CHARCOAL,
    )

    export(fig, "fig_vmod_margin_teacher", input_paths)


def confirmed_boundary_figure() -> None:
    _, main_run, _, _ = selected_runs()
    path_file = PROTOCOL / "capacity_path.csv"
    boundary_file = main_run / "confirmation_boundary_summary.json"
    conventional_file = main_run / "conventional_capacity_boundaries" / "summary.json"
    baselines_file = main_run / "final_baselines" / "summary.json"
    path = pd.read_csv(require(path_file)).sort_values("path_index")
    boundary = read_json(boundary_file)
    conventional = read_json(conventional_file)
    baselines = read_json(baselines_file)
    points = {row["role"]: row for row in boundary["points"]}
    selected = points["selected"]
    rejected = points["adjacent_rejected"]
    selected_label = str(selected["capacity_label"])
    vmod_daily_path = main_run / "confirmation" / selected_label / "daily.csv"
    pilot_daily_path = (
        main_run
        / "conventional_capacity_boundaries"
        / selected_label
        / "confirmation_refined"
        / "pilot_droop"
        / "daily.csv"
    )
    vmod_daily = pd.read_csv(require(vmod_daily_path))
    pilot_daily = pd.read_csv(require(pilot_daily_path))[
        ["day", "line_loss_mwh", "absolute_reactive_throughput_mvarh"]
    ].rename(
        columns={
            "line_loss_mwh": "pilot_line_loss_mwh",
            "absolute_reactive_throughput_mvarh": "pilot_reactive_throughput_mvarh",
        }
    )
    paired_daily = vmod_daily.merge(pilot_daily, on="day", validate="many_to_one")
    paired_daily = paired_daily.rename(
        columns={
            "line_loss_mwh": "vmod_line_loss_mwh",
            "absolute_reactive_throughput_mvarh": "vmod_reactive_throughput_mvarh",
        }
    )
    if len(paired_daily) != 1745:
        raise RuntimeError("Figure 3 requires all 1,745 seed--day confirmation pairs")
    loss_better = paired_daily.vmod_line_loss_mwh < paired_daily.pilot_line_loss_mwh
    throughput_better = (
        paired_daily.vmod_reactive_throughput_mvarh
        < paired_daily.pilot_reactive_throughput_mvarh
    )
    if not (loss_better.all() and throughput_better.all()):
        raise RuntimeError("Figure 3 paired-dominance claim is not supported by the data")
    paired_daily["line_loss_reduction_mwh"] = (
        paired_daily.pilot_line_loss_mwh - paired_daily.vmod_line_loss_mwh
    )
    paired_daily["reactive_throughput_reduction_mvarh"] = (
        paired_daily.pilot_reactive_throughput_mvarh
        - paired_daily.vmod_reactive_throughput_mvarh
    )
    figure_data_dir = FINAL_ROOT / "figure_data"
    figure_data_dir.mkdir(parents=True, exist_ok=True)
    paired_data_path = figure_data_dir / "figure3_paired_daily_efficiency.csv"
    paired_daily[
        [
            "seed",
            "day",
            "pilot_line_loss_mwh",
            "vmod_line_loss_mwh",
            "pilot_reactive_throughput_mvarh",
            "vmod_reactive_throughput_mvarh",
            "line_loss_reduction_mwh",
            "reactive_throughput_reduction_mvarh",
        ]
    ].to_csv(paired_data_path, index=False)
    conventional_by_method = {
        row["method"]: row for row in conventional.get("confirmation", [])
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.20, 5.10), layout="constrained")

    ax = axes[0, 0]
    ax.plot(
        path["pv_inverter_nameplate_mva"],
        path["svc_nameplate_mvar"],
        color=LIGHT_GREY,
        linewidth=1.4,
        zorder=1,
    )
    ax.scatter(
        path["pv_inverter_nameplate_mva"],
        path["svc_nameplate_mvar"],
        color="white",
        edgecolor=CHARCOAL,
        s=28,
        zorder=2,
    )
    for row in path.itertuples(index=False):
        ax.annotate(
            f"P{int(row.path_index):02d}",
            (row.pv_inverter_nameplate_mva, row.svc_nameplate_mvar),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=6.2,
            color=GREY,
        )
    ax.scatter(
        rejected["pv_inverter_nameplate_mva"],
        rejected["svc_nameplate_mvar"],
        marker="X",
        s=75,
        color=RED,
        edgecolor=CHARCOAL,
        linewidth=0.8,
        label="VMOD adjacent rejected",
        zorder=4,
    )
    ax.scatter(
        selected["pv_inverter_nameplate_mva"],
        selected["svc_nameplate_mvar"],
        marker="D",
        s=64,
        color=GREEN,
        edgecolor=CHARCOAL,
        linewidth=0.8,
        label="VMOD selected",
        zorder=4,
    )
    droop = conventional_by_method.get("pilot_droop", {})
    if droop.get("boundary_confirmed"):
        index = int(droop["selected_path_index"])
        row = path.loc[path.path_index == index].iloc[0]
        ax.scatter(
            row.pv_inverter_nameplate_mva,
            row.svc_nameplate_mvar,
            marker="s",
            s=60,
            color=BLUE,
            edgecolor=CHARCOAL,
            linewidth=0.8,
            label="Pilot droop selected",
            zorder=4,
        )
    ax.set_xlabel("PV-inverter nameplate (MVA)")
    ax.set_ylabel("SVC nameplate (MVAr)")
    panel_label(ax, "a", "Physical capacity path")
    tidy_axis(ax)
    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#9CA3AF",
        fontsize=6.2,
    )

    ax = axes[0, 1]
    selected_rows = {int(row["seed"]): row for row in selected["seed_summaries"]}
    rejected_rows = {int(row["seed"]): row for row in rejected["seed_summaries"]}
    seeds = sorted(selected_rows)
    x = np.arange(len(seeds))
    width = 0.36
    selected_events = [int(selected_rows[seed]["event_days"]) for seed in seeds]
    rejected_events = [int(rejected_rows[seed]["event_days"]) for seed in seeds]
    ax.bar(
        x - width / 2,
        selected_events,
        width,
        color=GREEN,
        edgecolor=CHARCOAL,
        linewidth=0.5,
        label=f"Selected P{int(selected['path_index']):02d}",
    )
    ax.bar(
        x + width / 2,
        rejected_events,
        width,
        color="#E6B8B3",
        edgecolor=CHARCOAL,
        linewidth=0.5,
        label=f"Rejected P{int(rejected['path_index']):02d}",
    )
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set_xlabel("Training seed")
    confirmation_days = int(boundary["confirmation_days_per_seed"])
    ax.set_ylabel(f"Event-days / {confirmation_days}")
    panel_label(ax, "b", "Seed-level confirmation")
    tidy_axis(ax)
    ax2 = ax.twinx()
    selected_upper = [
        100 * float(selected_rows[seed]["one_sided_clopper_pearson_upper_95"])
        for seed in seeds
    ]
    rejected_upper = [
        100 * float(rejected_rows[seed]["one_sided_clopper_pearson_upper_95"])
        for seed in seeds
    ]
    ax2.plot(x, selected_upper, color=GREEN, marker="o", markersize=3.5, linewidth=1.0)
    ax2.plot(x, rejected_upper, color=RED, marker="o", markersize=3.5, linewidth=1.0)
    ax2.axhline(1.0, color=CHARCOAL, linestyle="--", linewidth=0.8)
    ax2.set_ylabel("One-sided 95% upper bound (%)")
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#9CA3AF",
        fontsize=6.2,
    )

    def reduction_distribution(
        ax: plt.Axes,
        column: str,
        unit: str,
        letter: str,
        title: str,
    ) -> None:
        grouped = list(paired_daily.groupby("seed", sort=True))
        distributions = [group[column].to_numpy(dtype=float) for _, group in grouped]
        positions = np.arange(1, len(grouped) + 1)
        violins = ax.violinplot(
            distributions,
            positions=positions,
            widths=0.78,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, colour in zip(violins["bodies"], SEED_COLOURS):
            body.set_facecolor(mpl.colors.to_rgba(colour, 0.38))
            body.set_edgecolor(colour)
            body.set_linewidth(0.9)
            body.set_alpha(1.0)
        boxes = ax.boxplot(
            distributions,
            positions=positions,
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "white", "edgecolor": CHARCOAL, "linewidth": 0.8},
            medianprops={"color": CHARCOAL, "linewidth": 1.2},
            whiskerprops={"color": CHARCOAL, "linewidth": 0.8},
            capprops={"color": CHARCOAL, "linewidth": 0.8},
        )
        for box, colour in zip(boxes["boxes"], SEED_COLOURS):
            box.set_edgecolor(colour)
        means = [float(np.mean(values)) for values in distributions]
        ax.scatter(
            positions,
            means,
            marker="D",
            s=24,
            color=ORANGE,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        all_values = paired_daily[column].to_numpy(dtype=float)
        lower = float(all_values.min())
        upper = float(all_values.max())
        padding = 0.08 * (upper - lower)
        ax.set_ylim(lower - padding, upper + padding)
        ax.set_xticks(positions, [str(int(seed)) for seed, _ in grouped])
        ax.set_xlabel("Training seed")
        ax.set_ylabel(f"Reduction ({unit})")
        panel_label(ax, letter, title)
        tidy_axis(ax)
        ax.text(
            0.04,
            0.95,
            "1,745 / 1,745 reductions > 0",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            fontweight="bold",
            color=GREEN,
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": GREEN},
        )
        ax.text(
            0.96,
            0.05,
            f"Overall mean: {all_values.mean():.3f} {unit}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color=ORANGE,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": ORANGE},
        )

    reduction_distribution(
        axes[1, 0],
        "line_loss_reduction_mwh",
        "MWh/day",
        "c",
        "Daily line-loss reduction",
    )
    reduction_distribution(
        axes[1, 1],
        "reactive_throughput_reduction_mvarh",
        "MVArh/day",
        "d",
        "Daily reactive-throughput reduction",
    )

    export(
        fig,
        "fig_vmod_confirmed_boundary",
        [
            path_file,
            boundary_file,
            conventional_file,
            baselines_file,
            vmod_daily_path,
            pilot_daily_path,
            paired_data_path,
        ],
    )


def capacity_tradeoff_figure() -> None:
    _, main_run, _, _ = selected_runs()
    path_file = PROTOCOL / "capacity_path.csv"
    baselines_file = main_run / "final_baselines" / "summary.json"
    boundary_file = main_run / "confirmation_boundary_summary.json"
    path = pd.read_csv(require(path_file)).sort_values("path_index")
    baselines = read_json(baselines_file)
    boundary = read_json(boundary_file)
    selected = next(row for row in boundary["points"] if row["role"] == "selected")
    rejected = next(
        row for row in boundary["points"] if row["role"] == "adjacent_rejected"
    )
    opf_dir = Path(baselines["ac_opf_path_selection_dir"])
    opf_episodes_path = opf_dir / "episodes.csv"
    opf_summary_path = opf_dir / "summary.json"
    opf_episodes = pd.read_csv(require(opf_episodes_path)).rename(
        columns={"day_index": "day", "daily_loss_mwh": "opf_loss_mwh"}
    )
    opf_summary = read_json(opf_summary_path)
    if opf_episodes.candidate_id.nunique() != len(path):
        raise RuntimeError("Figure 4 requires AC-OPF results for every path point")
    matched_days = all_capacity_feasible_days(opf_episodes, len(path))
    expected_matched_days = int(opf_summary["all_capacity_converged_matched_days"])
    if len(matched_days) != expected_matched_days or not len(matched_days):
        raise RuntimeError(
            "All-capacity feasible AC-OPF support does not agree with its summary"
        )
    opf_episodes = opf_episodes.loc[opf_episodes.day.isin(matched_days)].copy()

    loss_rows = []
    knee_rows = []
    input_paths = [
        path_file,
        baselines_file,
        boundary_file,
        opf_episodes_path,
        opf_summary_path,
    ]
    for point in path.itertuples(index=False):
        summary_path = main_run / "selection" / str(point.capacity_label) / "summary.json"
        daily_path = main_run / "selection" / str(point.capacity_label) / "daily.csv"
        summary = read_json(summary_path)
        daily = pd.read_csv(require(daily_path))
        input_paths.extend([summary_path, daily_path])
        opf_point = opf_episodes.loc[
            opf_episodes.candidate_id == int(point.path_index),
            ["day", "opf_loss_mwh"],
        ]
        for seed, group in daily.groupby("seed", sort=True):
            paired = group.loc[group.day.isin(matched_days)].merge(
                opf_point, on="day", validate="one_to_one"
            )
            if len(paired) != expected_matched_days:
                raise RuntimeError(
                    "Figure 4 requires the same all-capacity feasible days for "
                    "every VMOD seed and capacity point"
                )
            loss_rows.append(
                {
                    "path_index": int(point.path_index),
                    "seed": int(seed),
                    "vmod_loss_mwh": float(paired.line_loss_mwh.mean()),
                    "opf_loss_mwh": float(paired.opf_loss_mwh.mean()),
                    "regret_mwh": float(
                        (paired.line_loss_mwh - paired.opf_loss_mwh).mean()
                    ),
                    "paired_days": int(len(paired)),
                }
            )
        knee_rows.append(
            {
                "path_index": int(point.path_index),
                "capacity_label": str(point.capacity_label),
                "total_event_days": int(summary["total_event_days"]),
                "mean_daily_line_loss_mwh": float(summary["mean_seed_loss_mwh"]),
            }
        )
    losses = pd.DataFrame(loss_rows)
    knee = pd.DataFrame(knee_rows).sort_values("path_index")
    figure_data_dir = FINAL_ROOT / "figure_data"
    figure_data_dir.mkdir(parents=True, exist_ok=True)
    loss_data_path = figure_data_dir / "figure4_seed_path_loss_regret.csv"
    knee_data_path = figure_data_dir / "figure4_capacity_efficiency_knee.csv"
    losses.to_csv(loss_data_path, index=False)
    knee.to_csv(knee_data_path, index=False)
    input_paths.extend([loss_data_path, knee_data_path])

    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.70), layout="constrained")
    x = path.path_index.to_numpy(dtype=int)
    labels = [f"P{value:02d}" for value in x]

    ax = axes[0]
    opf_means = losses.groupby("path_index").opf_loss_mwh.mean()
    ax.plot(x, opf_means.reindex(x), color=CHARCOAL, marker="D", linewidth=1.5, label="AC-OPF")
    for colour, (seed, group) in zip(SEED_COLOURS, losses.groupby("seed", sort=True)):
        group = group.sort_values("path_index")
        ax.plot(
            group.path_index,
            group.vmod_loss_mwh,
            color=colour,
            marker="o",
            markersize=3.0,
            linewidth=0.9,
            alpha=0.9,
            label=f"Seed {int(seed)}",
        )
    ax.set_xticks(x, labels, rotation=45)
    ax.set_xlabel("Capacity point")
    ax.set_ylabel("Mean daily line loss (MWh/day)")
    panel_label(ax, "a", "Matched feasible loss")
    ax.text(
        0.03,
        0.04,
        f"$n={expected_matched_days}$ common days",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=CHARCOAL,
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#9CA3AF"},
    )
    tidy_axis(ax)
    ax.legend(
        loc="upper right",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#9CA3AF",
        fontsize=5.8,
    )

    ax = axes[1]
    for colour, (seed, group) in zip(SEED_COLOURS, losses.groupby("seed", sort=True)):
        group = group.sort_values("path_index")
        ax.plot(
            group.path_index,
            group.regret_mwh,
            color=colour,
            marker="o",
            markersize=3.0,
            linewidth=0.9,
        )
    ax.axhline(0, color=CHARCOAL, linewidth=0.9)
    ax.set_xticks(x, labels, rotation=45)
    ax.set_xlabel("Capacity point")
    ax.set_ylabel("VMOD minus AC-OPF loss (MWh/day)")
    panel_label(ax, "b", "Matched controller regret")
    tidy_axis(ax)

    ax = axes[2]
    event_days = knee.set_index("path_index").total_event_days.reindex(x)
    mean_loss = knee.set_index("path_index").mean_daily_line_loss_mwh.reindex(x)
    bar_colours = [
        mpl.colors.to_rgba(RED, 0.68) if value > 0 else mpl.colors.to_rgba(GREEN, 0.25)
        for value in event_days
    ]
    ax.bar(
        x,
        event_days,
        color=bar_colours,
        edgecolor=[RED if value > 0 else GREEN for value in event_days],
        linewidth=0.8,
        label="Event-days",
    )
    zero_mask = event_days.to_numpy() == 0
    ax.scatter(x[zero_mask], np.zeros(zero_mask.sum()), marker="D", s=28, color=GREEN, zorder=4)
    ax.set_xticks(x, labels, rotation=45)
    ax.set_xlabel("Capacity point")
    ax.set_ylabel("Event-days across five actors", color=RED)
    ax.tick_params(axis="y", labelcolor=RED)
    ax2 = ax.twinx()
    ax2.plot(x, mean_loss, color=ORANGE, marker="o", markersize=4.0, linewidth=1.5, label="Line loss")
    ax2.set_ylabel("Mean daily line loss (MWh/day)", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    panel_label(ax, "c", "Capacity-efficiency knee")
    tidy_axis(ax)
    for point, text_value, colour in (
        (int(rejected["path_index"]), "Rejected", RED),
        (int(selected["path_index"]), "Selected", GREEN),
    ):
        ax.axvline(point, color=colour, linestyle="--", linewidth=0.9)
        ax.text(
            point + 0.08,
            0.70 if text_value == "Selected" else 0.84,
            text_value,
            transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
            color=CHARCOAL,
            fontsize=6.5,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": colour},
        )
    rejected_index = int(rejected["path_index"])
    selected_index = int(selected["path_index"])
    rejected_loss = float(mean_loss.loc[rejected_index])
    selected_loss = float(mean_loss.loc[selected_index])
    endpoint_loss = float(mean_loss.loc[int(x[-1])])
    first_drop = 100.0 * (selected_loss / rejected_loss - 1.0)
    tail_drop = 100.0 * (endpoint_loss / selected_loss - 1.0)
    ax2.annotate(
        f"P02→P03: {first_drop:.2f}%",
        xy=(selected_index, selected_loss),
        xytext=(selected_index + 0.7, selected_loss + 0.42),
        fontsize=6.4,
        fontweight="bold",
        color=ORANGE,
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 0.8},
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": ORANGE},
    )
    ax2.text(
        0.98,
        0.96,
        f"P03→P08: {tail_drop:.2f}%",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=6.4,
        fontweight="bold",
        color=CHARCOAL,
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": LIGHT_GREY},
    )

    export(fig, "fig_vmod_capacity_tradeoffs", input_paths)


def robustness_portability_runtime_figure() -> None:
    _, main_run, env69_run, _ = selected_runs()
    shift_file = main_run / "operational_shift" / "study_summary.json"
    main_boundary_file = main_run / "confirmation_boundary_summary.json"
    env69_boundary_file = env69_run / "confirmation_boundary_summary.json"
    runtime_file = main_run / "runtime_benchmark" / "summary.json"
    runtime_steps_file = main_run / "runtime_benchmark" / "steps.csv"
    shift = read_json(shift_file)
    main_boundary = read_json(main_boundary_file)
    env69_boundary = read_json(env69_boundary_file)
    runtime = read_json(runtime_file)
    runtime_steps = pd.read_csv(require(runtime_steps_file))
    env69_path_file = DEVELOPMENT_PROTOCOL_69 / "capacity_path.csv"
    env69_path = pd.read_csv(require(env69_path_file))

    levels = pd.DataFrame(shift["levels"])
    labels = [
        value.replace("design_topology_a", "design + topology A")
        .replace("delay_1_step", "delay 1")
        .replace("delay_4_steps", "delay 4")
        .replace("topology_a", "topology A")
        .replace("topology_b", "topology B")
        for value in levels.level
    ]
    rates = 100 * levels.plot_mean_event_rate.to_numpy(dtype=float)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.20, 3.20),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.42, 1.12, 1.20]},
    )

    ax = axes[0]
    y = np.arange(len(labels))
    low = 100 * levels.plot_event_rate_ci95_low.to_numpy(dtype=float)
    high = 100 * levels.plot_event_rate_ci95_high.to_numpy(dtype=float)
    xerr = np.vstack([rates - low, high - rates])
    colours = [
        GREEN if value <= 0.1 else ORANGE if value < 5.0 else RED
        for value in rates
    ]
    ax.errorbar(
        rates,
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#6B7280",
        elinewidth=0.8,
        capsize=2.0,
        zorder=1,
    )
    ax.scatter(rates, y, c=colours, edgecolor=CHARCOAL, linewidth=0.5, s=32, zorder=2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Event rate (%)")
    right = max(float(high.max()), float(rates.max()), 1.0)
    ax.set_xlim(-0.7, right * 1.23 + 0.7)
    for idx, row in levels.iterrows():
        low = 100 * float(row.plot_event_rate_ci95_low)
        high = 100 * float(row.plot_event_rate_ci95_high)
        ax.text(
            max(high, 100 * float(row.plot_mean_event_rate)) + 0.5,
            idx,
            f"{100 * float(row.plot_mean_event_rate):.1f}%",
            ha="left",
            va="center",
            fontsize=5.5,
            color=CHARCOAL,
        )
    ax.axhline(3.5, color=LIGHT_GREY, linewidth=0.8)
    ax.axhline(6.5, color=LIGHT_GREY, linewidth=0.8)
    panel_label(ax, "a", "Empirical work domain")
    tidy_axis(ax, "x")

    ax = axes[1]
    systems = [("IEEE 33-bus", main_boundary), ("IEEE 69-bus", env69_boundary)]
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.45, 1.45)
    rejected_x = 0.42
    selected_x = 0.82
    for row_index, (name, boundary) in enumerate(systems):
        y_value = 1 - row_index
        selected = next(
            (row for row in boundary.get("points", []) if row["role"] == "selected"),
            None,
        )
        rejected = next(
            (
                row
                for row in boundary.get("points", [])
                if row["role"] == "adjacent_rejected"
            ),
            None,
        )
        if selected is None or rejected is None:
            ax.text(
                0.08,
                y_value,
                "No development point passed (15 tested)",
                color=RED,
                va="center",
                ha="left",
                fontsize=5.3,
                bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": RED},
            )
            continue
        card = FancyBboxPatch(
            (0.02, y_value - 0.31),
            0.96,
            0.62,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            facecolor=mpl.colors.to_rgba(BLUE if row_index == 0 else TEAL, 0.055),
            edgecolor=LIGHT_GREY,
            linewidth=0.8,
            zorder=0,
        )
        ax.add_patch(card)
        ax.text(0.06, y_value, name, ha="left", va="center", fontsize=7.0, fontweight="bold")
        ax.add_patch(
            FancyArrowPatch(
                (rejected_x + 0.04, y_value),
                (selected_x - 0.05, y_value),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.1,
                color=CHARCOAL,
                zorder=1,
            )
        )
        ax.scatter(
            rejected_x,
            y_value,
            marker="X",
            s=65,
            color=RED,
            edgecolor=CHARCOAL,
            linewidth=0.7,
        )
        ax.scatter(
            selected_x,
            y_value,
            marker="D",
            s=60,
            color=GREEN,
            edgecolor=CHARCOAL,
            linewidth=0.7,
        )
        ax.text(
            rejected_x,
            y_value + 0.17,
            f"P{int(rejected['path_index']):02d}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
            color=RED,
        )
        ax.text(
            selected_x,
            y_value + 0.17,
            f"P{int(selected['path_index']):02d}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
            color=GREEN,
        )
        selected_days = int(boundary.get("confirmation_days_per_seed", 0))
        selected_note = f"0/{selected_days} each"
        rejected_note = "rejected" if row_index == 0 else "1 event seed-day"
        ax.text(rejected_x, y_value - 0.18, rejected_note, ha="center", va="top", fontsize=5.8, color=RED)
        ax.text(selected_x, y_value - 0.18, selected_note, ha="center", va="top", fontsize=5.8, color=GREEN)
    ax.text(
        0.50,
        0.50,
        "feeder-specific adaptation",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.0,
        fontstyle="italic",
        color=GREY,
    )
    ax.axis("off")
    panel_label(ax, "b", "Adjacent boundaries")

    ax = axes[2]
    columns = ["actor_inference_ms", "ac_execution_ms", "total_ms"]
    labels = ["Actor", "AC plant", "Total"]
    colours = [ORANGE, CHARCOAL, BLUE]
    latency_values = [
        runtime_steps[column].to_numpy(dtype=float) for column in columns
    ]
    percentile_rows = [
        {
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "maximum": float(np.max(values)),
        }
        for values in latency_values
    ]
    y = np.arange(len(columns))
    for y_value, row, colour in zip(y, percentile_rows, colours):
        ax.hlines(y_value, row["p05"], row["p95"], color=colour, linewidth=3.0)
        ax.scatter(row["p50"], y_value, marker="o", s=28, color="white", edgecolor=colour, zorder=3)
        ax.scatter(row["p95"], y_value, marker="D", s=32, color=colour, edgecolor="white", linewidth=0.5, zorder=4)
        ax.scatter(row["maximum"], y_value, marker="^", s=34, color="white", edgecolor=colour, linewidth=1.0, zorder=4)
        label_above = y_value == len(percentile_rows) - 1
        ax.annotate(
            f"p95 {row['p95']:.2f}",
            (row["p95"], y_value),
            xytext=(5, 8 if label_above else -8),
            textcoords="offset points",
            ha="left",
            va="bottom" if label_above else "top",
            fontsize=5.8,
            color=colour,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms, log scale)")
    panel_label(ax, "c", "Online latency")
    tidy_axis(ax, "x")
    positive_values = np.concatenate([values[values > 0] for values in latency_values])
    ax.set_xlim(float(positive_values.min()) / 1.5, float(positive_values.max()) * 1.8)
    ax.legend(
        handles=[
            mpl.lines.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=CHARCOAL, label="Median"),
            mpl.lines.Line2D([], [], marker="D", linestyle="", color=CHARCOAL, label="95th percentile"),
            mpl.lines.Line2D([], [], marker="^", linestyle="", markerfacecolor="white", markeredgecolor=CHARCOAL, label="Maximum"),
        ],
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GREY,
        fontsize=5.8,
    )
    ax.text(
        0.98,
        0.64,
        "15-min interval: 900,000 ms\nTotal p95: 0.0023% of interval",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=GREY,
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": LIGHT_GREY},
    )

    export(
        fig,
        "fig_vmod_robustness_runtime",
        [
            shift_file,
            main_boundary_file,
            env69_boundary_file,
            runtime_file,
            runtime_steps_file,
            env69_path_file,
        ],
    )


def validation_evidence_figure() -> None:
    """Summarise robustness, learning stability, baselines and model sharing."""
    _, main_run, _, _ = selected_runs()
    shift_file = main_run / "operational_shift" / "study_summary.json"
    baseline_file = main_run / "final_baselines" / "summary.json"
    sensitivity_file = main_run / "sensitivity_qp_baseline" / "summary.json"
    conditioning_file = main_run / "conditioning_ablation" / "summary.json"
    conditioning_contrasts_file = (
        main_run / "conditioning_ablation" / "seed_level_contrasts.csv"
    )
    training_files = sorted((main_run / "policies").glob("training_seed*.csv"))
    if len(training_files) != 5:
        raise ValueError(f"Expected five training logs, found {len(training_files)}")

    shift = read_json(shift_file)
    baselines = read_json(baseline_file)
    sensitivity = read_json(sensitivity_file)
    conditioning = read_json(conditioning_file)
    contrasts = pd.read_csv(require(conditioning_contrasts_file))
    confirmation_contrasts = contrasts.loc[
        contrasts["split"] == "confirmation"
    ].copy()

    training = []
    for path in training_files:
        frame = pd.read_csv(require(path))
        frame["seed"] = int(path.stem.rsplit("seed", 1)[-1])
        training.append(frame)
    training = pd.concat(training, ignore_index=True)
    training_summary = (
        training.groupby("epoch", as_index=False)
        .agg(
            training_mean=("training_mse", "mean"),
            training_min=("training_mse", "min"),
            training_max=("training_mse", "max"),
            validation_mean=("validation_mse", "mean"),
            validation_min=("validation_mse", "min"),
            validation_max=("validation_mse", "max"),
        )
        .sort_values("epoch")
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.20, 5.25),
        layout="constrained",
        gridspec_kw={"height_ratios": [1.03, 1.0], "width_ratios": [1.08, 1.0]},
    )

    # (a) Empirical robustness domain.
    ax = axes[0, 0]
    levels = pd.DataFrame(shift["levels"])
    labels = [
        value.replace("design_topology_a", "design + topology A")
        .replace("delay_1_step", "delay 1")
        .replace("delay_4_steps", "delay 4")
        .replace("topology_a", "topology A")
        .replace("topology_b", "topology B")
        for value in levels.level
    ]
    rates = 100 * levels.plot_mean_event_rate.to_numpy(dtype=float)
    low = 100 * levels.plot_event_rate_ci95_low.to_numpy(dtype=float)
    high = 100 * levels.plot_event_rate_ci95_high.to_numpy(dtype=float)
    xerr = np.vstack([rates - low, high - rates])
    y = np.arange(len(labels))
    colours = [
        GREEN if value <= 0.1 else ORANGE if value < 5.0 else RED
        for value in rates
    ]
    ax.errorbar(
        rates,
        y,
        xerr=xerr,
        fmt="none",
        ecolor=GREY,
        elinewidth=0.8,
        capsize=2.0,
        zorder=1,
    )
    ax.scatter(
        rates,
        y,
        c=colours,
        edgecolor=CHARCOAL,
        linewidth=0.5,
        s=30,
        zorder=2,
    )
    for index, (rate, upper) in enumerate(zip(rates, high)):
        ax.text(
            max(rate, upper) + 0.45,
            index,
            f"{rate:.1f}%",
            ha="left",
            va="center",
            fontsize=5.8,
            color=CHARCOAL,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Event rate (%)")
    ax.set_xlim(-0.7, max(float(high.max()), float(rates.max()), 1.0) * 1.22 + 0.7)
    ax.axhline(3.5, color=LIGHT_GREY, linewidth=0.8)
    ax.axhline(6.5, color=LIGHT_GREY, linewidth=0.8)
    panel_label(ax, "a", "Operating-shift response")
    tidy_axis(ax, "x")

    # (b) Five-seed supervised fitting dynamics.
    ax = axes[0, 1]
    epoch = training_summary["epoch"].to_numpy(dtype=float)
    for prefix, colour, label, linestyle in (
        ("training", BLUE, "Training", "-"),
        ("validation", ORANGE, "Validation", "--"),
    ):
        mean = training_summary[f"{prefix}_mean"].to_numpy(dtype=float)
        lower = training_summary[f"{prefix}_min"].to_numpy(dtype=float)
        upper = training_summary[f"{prefix}_max"].to_numpy(dtype=float)
        ax.fill_between(epoch, lower, upper, color=colour, alpha=0.13, linewidth=0)
        ax.plot(epoch, mean, color=colour, linewidth=1.35, linestyle=linestyle, label=label)
    final_validation = float(training_summary["validation_mean"].iloc[-1])
    ax.annotate(
        f"Final validation mean\n{final_validation:.2e}",
        (epoch[-1], final_validation),
        xytext=(-58, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": ORANGE, "linewidth": 0.7},
        fontsize=6.0,
        color=ORANGE,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": LIGHT_GREY},
    )
    ax.set_yscale("log")
    ax.set_xlim(1, float(epoch[-1]))
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Action-imitation MSE")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    panel_label(ax, "b", "Five-seed fitting convergence")
    tidy_axis(ax, "both")

    # (c) Confirmation outcomes for the conventional controls and VMOD.
    ax = axes[1, 0]
    confirmation = {
        row["baseline"]: row
        for row in baselines["comparisons"]
        if row["split"] == "confirmation"
    }
    boundary_points = read_json(
        main_run / "confirmation_boundary_summary.json"
    )["points"]
    selected_label = str(
        next(row for row in boundary_points if row["role"] == "selected")[
            "capacity_label"
        ]
    )
    qp_confirmation = next(
        row
        for row in sensitivity["results"]
        if row["capacity_label"] == selected_label
        and row["split"] == "confirmation"
    )
    comparison_rows = [
        (
            "VMOD",
            float(baselines["vmod_confirmation"]["mean_daily_loss_mwh"]),
            int(baselines["vmod_confirmation"]["event_days"]),
            GREEN,
            "D",
        ),
        (
            "Sensitivity QP",
            float(qp_confirmation["mean_daily_loss_mwh"]),
            int(qp_confirmation["event_days"]),
            TEAL,
            "^",
        ),
        (
            "Pilot droop",
            float(confirmation["pilot_droop"]["mean_daily_loss_mwh"]),
            int(confirmation["pilot_droop"]["event_days"]),
            BLUE,
            "o",
        ),
        (
            "Local droop",
            float(confirmation["droop"]["mean_daily_loss_mwh"]),
            int(confirmation["droop"]["event_days"]),
            ORANGE,
            "s",
        ),
        (
            "No control",
            float(confirmation["no_control"]["mean_daily_loss_mwh"]),
            int(confirmation["no_control"]["event_days"]),
            RED,
            "X",
        ),
    ]
    ax.axhspan(0, 0.01 * baselines["confirmation_days"], color=GREEN, alpha=0.08, zorder=0)
    ax.axhline(
        0.01 * baselines["confirmation_days"],
        color=GREEN,
        linewidth=0.8,
        linestyle="--",
        zorder=0,
    )
    annotation_offsets = {
        "VMOD": (5, 10),
        "Pilot droop": (-4, 10),
        "Local droop": (6, -2),
        "No control": (-5, -13),
    }
    for name, loss, event_days, colour, marker in comparison_rows:
        ax.scatter(
            loss,
            event_days,
            s=58,
            color=colour,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        dx, dy = annotation_offsets[name]
        ax.annotate(
            name,
            (loss, event_days),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.5,
            fontweight="bold" if name == "VMOD" else "normal",
            color=colour,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
        )
    ax.annotate(
        "preferred",
        xy=(1.53, 18),
        xytext=(1.78, 100),
        arrowprops={"arrowstyle": "-|>", "color": GREY, "linewidth": 0.8},
        fontsize=6.0,
        color=GREY,
        ha="center",
    )
    ax.set_xlim(1.45, 2.75)
    ax.set_ylim(-18, 370)
    ax.set_xlabel("Mean daily line loss (MWh/day)")
    ax.set_ylabel(f"Event-days / {int(baselines['confirmation_days'])}")
    panel_label(ax, "c", "Confirmation baseline comparison")
    tidy_axis(ax, "both")

    # (d) Capacity-conditioned actor versus separately fitted capacity-specific actors.
    ax = axes[1, 1]
    confirmation_contrasts = confirmation_contrasts.sort_values("seed")
    seed_values = confirmation_contrasts["shared_minus_fixed_loss_mwh"].to_numpy(dtype=float)
    seed_labels = [f"Seed {int(value)}" for value in confirmation_contrasts["seed"]]
    mean_value = float(conditioning["shared_minus_fixed_loss_mean_mwh"])
    ci_low = float(conditioning["seed_level_t_ci95_low_mwh"])
    ci_high = float(conditioning["seed_level_t_ci95_high_mwh"])
    y_seed = np.arange(len(seed_values))
    ax.axvline(0, color=CHARCOAL, linewidth=0.8, linestyle="--")
    ax.scatter(
        seed_values,
        y_seed,
        c=SEED_COLOURS,
        s=32,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    mean_y = len(seed_values) + 0.45
    ax.errorbar(
        mean_value,
        mean_y,
        xerr=np.array([[mean_value - ci_low], [ci_high - mean_value]]),
        fmt="D",
        color=TEAL,
        ecolor=TEAL,
        elinewidth=1.5,
        capsize=3,
        markersize=5,
        markeredgecolor="white",
        zorder=4,
    )
    ax.axhline(len(seed_values) - 0.35, color=LIGHT_GREY, linewidth=0.8)
    ax.set_yticks([*y_seed, mean_y], [*seed_labels, "Mean (95% CI)"])
    ax.invert_yaxis()
    limit = max(abs(ci_low), abs(ci_high), float(np.max(np.abs(seed_values)))) * 1.28
    ax.set_xlim(-limit, limit)
    ax.set_xticks([-0.005, 0.0, 0.005])
    ax.set_xlabel("Shared - fixed loss (MWh/day)")
    panel_label(ax, "d", "Capacity-conditioning ablation")
    tidy_axis(ax, "x")

    source_rows: list[dict] = []
    for _, row in levels.iterrows():
        source_rows.append(
            {
                "panel": "a",
                "series": row["level"],
                "x": 100 * float(row["plot_mean_event_rate"]),
                "lower": 100 * float(row["plot_event_rate_ci95_low"]),
                "upper": 100 * float(row["plot_event_rate_ci95_high"]),
                "unit": "percent event rate",
            }
        )
    for _, row in training_summary.iterrows():
        for prefix in ("training", "validation"):
            source_rows.append(
                {
                    "panel": "b",
                    "series": prefix,
                    "x": float(row["epoch"]),
                    "y": float(row[f"{prefix}_mean"]),
                    "lower": float(row[f"{prefix}_min"]),
                    "upper": float(row[f"{prefix}_max"]),
                    "unit": "action-imitation MSE",
                }
            )
    for name, loss, event_days, _, _ in comparison_rows:
        source_rows.append(
            {
                "panel": "c",
                "series": name,
                "x": loss,
                "y": event_days,
                "unit": "MWh/day; event-days",
            }
        )
    for seed, value in zip(confirmation_contrasts["seed"], seed_values):
        source_rows.append(
            {
                "panel": "d",
                "series": f"seed {int(seed)}",
                "x": value,
                "unit": "shared-minus-fixed MWh/day",
            }
        )
    source_rows.append(
        {
            "panel": "d",
            "series": "mean",
            "x": mean_value,
            "lower": ci_low,
            "upper": ci_high,
            "unit": "shared-minus-fixed MWh/day",
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_csv(
        OUT / "fig_vmod_validation_evidence_source_data.csv", index=False
    )

    export(
        fig,
        "fig_vmod_validation_evidence",
        [
            shift_file,
            baseline_file,
            conditioning_file,
            conditioning_contrasts_file,
            *training_files,
        ],
    )


def split_validation_figures() -> None:
    """Place learning, baseline and robustness evidence beside their claims."""
    _, main_run, _, _ = selected_runs()
    shift_file = main_run / "operational_shift" / "study_summary.json"
    baseline_file = main_run / "final_baselines" / "summary.json"
    sensitivity_file = main_run / "sensitivity_qp_baseline" / "summary.json"
    conditioning_file = main_run / "conditioning_ablation" / "summary.json"
    conditioning_contrasts_file = (
        main_run / "conditioning_ablation" / "seed_level_contrasts.csv"
    )
    training_files = sorted((main_run / "policies").glob("training_seed*.csv"))
    if len(training_files) != 5:
        raise ValueError(f"Expected five training logs, found {len(training_files)}")

    shift = read_json(shift_file)
    baselines = read_json(baseline_file)
    sensitivity = read_json(sensitivity_file)
    conditioning = read_json(conditioning_file)
    contrasts = pd.read_csv(require(conditioning_contrasts_file))
    confirmation_contrasts = contrasts.loc[
        contrasts["split"] == "confirmation"
    ].copy()

    training_frames = []
    for path in training_files:
        frame = pd.read_csv(require(path))
        frame["seed"] = int(path.stem.rsplit("seed", 1)[-1])
        training_frames.append(frame)
    training = pd.concat(training_frames, ignore_index=True)
    training_summary = (
        training.groupby("epoch", as_index=False)
        .agg(
            training_mean=("training_mse", "mean"),
            training_min=("training_mse", "min"),
            training_max=("training_mse", "max"),
            validation_mean=("validation_mse", "mean"),
            validation_min=("validation_mse", "min"),
            validation_max=("validation_mse", "max"),
        )
        .sort_values("epoch")
    )

    # Learning dynamics and the model-sharing ablation answer one design question.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.20, 2.75),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.18, 1.0]},
    )
    ax = axes[0]
    epoch = training_summary["epoch"].to_numpy(dtype=float)
    for prefix, colour, label, linestyle in (
        ("training", BLUE, "Training", "-"),
        ("validation", ORANGE, "Validation", "--"),
    ):
        mean = training_summary[f"{prefix}_mean"].to_numpy(dtype=float)
        lower = training_summary[f"{prefix}_min"].to_numpy(dtype=float)
        upper = training_summary[f"{prefix}_max"].to_numpy(dtype=float)
        ax.fill_between(epoch, lower, upper, color=colour, alpha=0.13, linewidth=0)
        ax.plot(epoch, mean, color=colour, linewidth=1.35, linestyle=linestyle, label=label)
    final_validation = float(training_summary["validation_mean"].iloc[-1])
    ax.annotate(
        f"Final validation mean\n{final_validation:.2e}",
        (epoch[-1], final_validation),
        xytext=(-58, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": ORANGE, "linewidth": 0.7},
        fontsize=6.2,
        color=ORANGE,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": LIGHT_GREY},
    )
    ax.set_yscale("log")
    ax.set_xlim(1, float(epoch[-1]))
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Action-imitation MSE")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    panel_label(ax, "a", "Five-seed fitting convergence")
    tidy_axis(ax, "both")

    ax = axes[1]
    confirmation_contrasts = confirmation_contrasts.sort_values("seed")
    seed_values = confirmation_contrasts["shared_minus_fixed_loss_mwh"].to_numpy(dtype=float)
    seed_labels = [f"Seed {int(value)}" for value in confirmation_contrasts["seed"]]
    mean_value = float(conditioning["shared_minus_fixed_loss_mean_mwh"])
    ci_low = float(conditioning["seed_level_t_ci95_low_mwh"])
    ci_high = float(conditioning["seed_level_t_ci95_high_mwh"])
    y_seed = np.arange(len(seed_values))
    ax.axvline(0, color=CHARCOAL, linewidth=0.8, linestyle="--")
    ax.scatter(
        seed_values,
        y_seed,
        c=SEED_COLOURS,
        s=32,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    mean_y = len(seed_values) + 0.45
    ax.errorbar(
        mean_value,
        mean_y,
        xerr=np.array([[mean_value - ci_low], [ci_high - mean_value]]),
        fmt="D",
        color=TEAL,
        ecolor=TEAL,
        elinewidth=1.5,
        capsize=3,
        markersize=5,
        markeredgecolor="white",
        zorder=4,
    )
    ax.axhline(len(seed_values) - 0.35, color=LIGHT_GREY, linewidth=0.8)
    ax.set_yticks([*y_seed, mean_y], [*seed_labels, "Mean (95% CI)"])
    ax.invert_yaxis()
    limit = max(abs(ci_low), abs(ci_high), float(np.max(np.abs(seed_values)))) * 1.28
    ax.set_xlim(-limit, limit)
    ax.set_xticks([-0.005, 0.0, 0.005])
    ax.set_xlabel("Shared - fixed loss (MWh/day)")
    panel_label(ax, "b", "Capacity-conditioning ablation")
    tidy_axis(ax, "x")

    learning_rows: list[dict] = []
    for _, row in training_summary.iterrows():
        for prefix in ("training", "validation"):
            learning_rows.append(
                {
                    "panel": "a",
                    "series": prefix,
                    "x": float(row["epoch"]),
                    "y": float(row[f"{prefix}_mean"]),
                    "lower": float(row[f"{prefix}_min"]),
                    "upper": float(row[f"{prefix}_max"]),
                    "unit": "action-imitation MSE",
                }
            )
    for seed, value in zip(confirmation_contrasts["seed"], seed_values):
        learning_rows.append(
            {
                "panel": "b",
                "series": f"seed {int(seed)}",
                "x": value,
                "unit": "shared-minus-fixed MWh/day",
            }
        )
    learning_rows.append(
        {
            "panel": "b",
            "series": "mean",
            "x": mean_value,
            "lower": ci_low,
            "upper": ci_high,
            "unit": "shared-minus-fixed MWh/day",
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(learning_rows).to_csv(
        OUT / "fig_vmod_learning_ablation_source_data.csv", index=False
    )
    export(
        fig,
        "fig_vmod_learning_ablation",
        [conditioning_file, conditioning_contrasts_file, *training_files],
    )

    # Confirmation baselines answer the controller-comparison question.
    confirmation = {
        row["baseline"]: row
        for row in baselines["comparisons"]
        if row["split"] == "confirmation"
    }
    boundary_points = read_json(
        main_run / "confirmation_boundary_summary.json"
    )["points"]
    selected_label = str(
        next(row for row in boundary_points if row["role"] == "selected")[
            "capacity_label"
        ]
    )
    qp_confirmation = next(
        row
        for row in sensitivity["results"]
        if row["capacity_label"] == selected_label
        and row["split"] == "confirmation"
    )
    comparison_rows = [
        (
            "VMOD",
            float(baselines["vmod_confirmation"]["mean_daily_loss_mwh"]),
            int(baselines["vmod_confirmation"]["event_days"]),
            GREEN,
            "D",
        ),
        (
            "Sensitivity QP",
            float(qp_confirmation["mean_daily_loss_mwh"]),
            int(qp_confirmation["event_days"]),
            TEAL,
            "^",
        ),
        (
            "Pilot droop",
            float(confirmation["pilot_droop"]["mean_daily_loss_mwh"]),
            int(confirmation["pilot_droop"]["event_days"]),
            BLUE,
            "o",
        ),
        (
            "Local droop",
            float(confirmation["droop"]["mean_daily_loss_mwh"]),
            int(confirmation["droop"]["event_days"]),
            ORANGE,
            "s",
        ),
        (
            "No control",
            float(confirmation["no_control"]["mean_daily_loss_mwh"]),
            int(confirmation["no_control"]["event_days"]),
            RED,
            "X",
        ),
    ]
    fig, ax = plt.subplots(figsize=(5.05, 3.10), layout="constrained")
    ax.axhspan(0, 0.01 * baselines["confirmation_days"], color=GREEN, alpha=0.08, zorder=0)
    ax.axhline(
        0.01 * baselines["confirmation_days"],
        color=GREEN,
        linewidth=0.8,
        linestyle="--",
        zorder=0,
    )
    annotation_offsets = {
        "VMOD": (6, 11),
        "Sensitivity QP": (7, 11),
        "Pilot droop": (-5, 11),
        "Local droop": (7, -2),
        "No control": (-7, -13),
    }
    for name, loss, event_days, colour, marker in comparison_rows:
        ax.scatter(
            loss,
            event_days,
            s=66,
            color=colour,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        dx, dy = annotation_offsets[name]
        ax.annotate(
            name,
            (loss, event_days),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.0,
            fontweight="bold" if name == "VMOD" else "normal",
            color=colour,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
        )
    ax.annotate(
        "preferred",
        xy=(1.53, 18),
        xytext=(1.78, 100),
        arrowprops={"arrowstyle": "-|>", "color": GREY, "linewidth": 0.8},
        fontsize=6.5,
        color=GREY,
        ha="center",
    )
    ax.set_xlim(1.45, 2.75)
    ax.set_ylim(-18, 370)
    ax.set_xlabel("Mean daily line loss (MWh/day)")
    ax.set_ylabel(f"Event-days / {int(baselines['confirmation_days'])}")
    ax.set_title("Confirmation baseline comparison", loc="left")
    tidy_axis(ax, "both")
    pd.DataFrame(
        [
            {
                "controller": name,
                "mean_daily_line_loss_mwh": loss,
                "event_days": event_days,
            }
            for name, loss, event_days, _, _ in comparison_rows
        ]
    ).to_csv(OUT / "fig_vmod_baseline_comparison_source_data.csv", index=False)
    export(
        fig,
        "fig_vmod_baseline_comparison",
        [baseline_file, sensitivity_file],
    )

    # The operating-shift plot belongs with the empirical work-domain text.
    levels = pd.DataFrame(shift["levels"])
    labels = [
        value.replace("design_topology_a", "design + topology A")
        .replace("delay_1_step", "delay 1")
        .replace("delay_4_steps", "delay 4")
        .replace("topology_a", "topology A")
        .replace("topology_b", "topology B")
        for value in levels.level
    ]
    rates = 100 * levels.plot_mean_event_rate.to_numpy(dtype=float)
    low = 100 * levels.plot_event_rate_ci95_low.to_numpy(dtype=float)
    high = 100 * levels.plot_event_rate_ci95_high.to_numpy(dtype=float)
    xerr = np.vstack([rates - low, high - rates])
    y = np.arange(len(labels))
    colours = [
        GREEN if value <= 0.1 else ORANGE if value < 5.0 else RED
        for value in rates
    ]
    fig, ax = plt.subplots(figsize=(5.45, 3.05), layout="constrained")
    ax.errorbar(
        rates,
        y,
        xerr=xerr,
        fmt="none",
        ecolor=GREY,
        elinewidth=0.8,
        capsize=2.0,
        zorder=1,
    )
    ax.scatter(
        rates,
        y,
        c=colours,
        edgecolor=CHARCOAL,
        linewidth=0.5,
        s=34,
        zorder=2,
    )
    for index, (rate, upper) in enumerate(zip(rates, high)):
        ax.text(
            max(rate, upper) + 0.45,
            index,
            f"{rate:.1f}%",
            ha="left",
            va="center",
            fontsize=6.2,
            color=CHARCOAL,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Event rate (%)")
    ax.set_xlim(-0.7, max(float(high.max()), float(rates.max()), 1.0) * 1.22 + 0.7)
    ax.axhline(3.5, color=LIGHT_GREY, linewidth=0.8)
    ax.axhline(6.5, color=LIGHT_GREY, linewidth=0.8)
    ax.set_title("Operating-shift response", loc="left")
    tidy_axis(ax, "x")
    pd.DataFrame(
        {
            "condition": levels["level"],
            "mean_event_rate_percent": rates,
            "ci95_low_percent": low,
            "ci95_high_percent": high,
        }
    ).to_csv(OUT / "fig_vmod_operating_shifts_source_data.csv", index=False)
    export(fig, "fig_vmod_operating_shifts", [shift_file])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        default="protocol",
        help="Comma-separated: protocol,margin,boundary,tradeoff,validation,all",
    )
    args = parser.parse_args()
    requested = {item.strip().lower() for item in args.figures.split(",") if item.strip()}
    if "all" in requested:
        requested = {"protocol", "margin", "boundary", "tradeoff", "validation"}
    unknown = requested - {
        "protocol",
        "margin",
        "boundary",
        "tradeoff",
        "validation",
        "robustness",
    }
    if unknown:
        parser.error(f"Unknown figure group(s): {', '.join(sorted(unknown))}")
    configure_style()
    if "protocol" in requested:
        protocol_figure()
    if "margin" in requested:
        margin_and_teacher_figure()
    if "boundary" in requested:
        confirmed_boundary_figure()
    if "tradeoff" in requested:
        capacity_tradeoff_figure()
    if "validation" in requested or "robustness" in requested:
        split_validation_figures()


if __name__ == "__main__":
    main()
