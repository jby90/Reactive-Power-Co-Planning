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
    image = ax.imshow(np.asarray(event_matrix), cmap="YlOrRd", aspect="auto", vmin=0)
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
    ax.text(7.9, 32.6, "32-day gate", ha="right", va="bottom", color=RED, fontsize=7.3)
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
        0.04,
        0.06,
        f"Max positive increment: {max_increase:.2e} MWh/day\n"
        "Tolerance: $10^{-4}$ MWh/day + 0.1%",
        transform=ax.transAxes,
        fontsize=6.8,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#9CA3AF"},
    )

    export(fig, "fig_vmod_margin_teacher", input_paths)


def confirmed_boundary_figure() -> None:
    _, main_run, _, claim = selected_runs()
    path_file = PROTOCOL / "capacity_path.csv"
    boundary_file = main_run / "confirmation_boundary_summary.json"
    conventional_file = main_run / "conventional_capacity_boundaries" / "summary.json"
    path = pd.read_csv(require(path_file)).sort_values("path_index")
    boundary = read_json(boundary_file)
    conventional = read_json(conventional_file)
    points = {row["role"]: row for row in boundary["points"]}
    selected = points["selected"]
    rejected = points["adjacent_rejected"]
    conventional_by_method = {
        row["method"]: row for row in conventional.get("confirmation", [])
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.70), layout="constrained")

    ax = axes[0]
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
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#9CA3AF",
        fontsize=6.2,
    )

    ax = axes[1]
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

    ax = axes[2]
    methods = ["VMOD", "Pilot droop", "Local droop", "No control"]
    values: list[float] = [float(selected["path_index"])]
    colours = [ORANGE]
    for method, colour in (
        ("pilot_droop", BLUE),
        ("droop", TEAL),
        ("no_control", "#9CA3AF"),
    ):
        row = conventional_by_method.get(method, {})
        values.append(
            np.nan
            if not row.get("boundary_confirmed")
            else float(row["selected_path_index"])
        )
        colours.append(colour)
    y = np.arange(len(methods))
    ax.scatter(values, y, marker="D", s=65, c=colours, edgecolor=CHARCOAL, linewidth=0.8)
    ax.set_yticks(y, methods)
    ax.set_xticks(
        path.path_index,
        [f"P{int(value):02d}" for value in path.path_index],
        rotation=45,
        ha="right",
    )
    ax.set_xlim(-0.5, float(path.path_index.max()) + 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Lowest confirmed tested point")
    panel_label(ax, "c", "Independent method boundaries")
    tidy_axis(ax, "x")
    if claim["permitted_claims"].get("controller_reduces_required_capacity_vs_tuned_droop"):
        difference = int(droop["selected_path_index"]) - int(selected["path_index"])
        ax.annotate(
            f"{difference} path step{'s' if difference != 1 else ''} lower",
            xy=(selected["path_index"], 0),
            xytext=(droop["selected_path_index"], 1),
            arrowprops={"arrowstyle": "<->", "color": CHARCOAL, "lw": 1.0},
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#9CA3AF"},
        )
    elif claim["permitted_claims"].get(
        "no_greater_capacity_and_lower_loss_vs_pilot_droop"
    ) and values[0] == values[1]:
        ax.text(
            0.5,
            0.08,
            "Same confirmed boundary; lower VMOD loss",
            transform=ax.transAxes,
            ha="center",
            fontsize=7.2,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#9CA3AF"},
        )
    elif np.isnan(values[1]):
        ax.text(
            0.5,
            0.08,
            "No confirmed pilot-droop boundary",
            transform=ax.transAxes,
            ha="center",
            fontsize=7.2,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#9CA3AF"},
        )

    export(
        fig,
        "fig_vmod_confirmed_boundary",
        [path_file, boundary_file, conventional_file, FINAL_ROOT / "claim_gate_summary.json"],
    )


def capacity_tradeoff_figure() -> None:
    _, main_run, _, claim = selected_runs()
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
    actuation_rows = []
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
        for row in summary["seed_summaries"]:
            actuation_rows.append(
                {
                    "path_index": int(point.path_index),
                    "seed": int(row["seed"]),
                    "throughput_mvarh": float(
                        row["mean_daily_absolute_reactive_throughput_mvarh"]
                    ),
                    "saturation_percent": 100
                    * float(row["normalized_action_saturation_fraction"]),
                }
            )
    losses = pd.DataFrame(loss_rows)
    actuation = pd.DataFrame(actuation_rows)
    figure_data_dir = FINAL_ROOT / "figure_data"
    figure_data_dir.mkdir(parents=True, exist_ok=True)
    loss_data_path = figure_data_dir / "figure4_seed_path_loss_regret.csv"
    actuation_data_path = figure_data_dir / "figure4_seed_path_actuation.csv"
    losses.to_csv(loss_data_path, index=False)
    actuation.to_csv(actuation_data_path, index=False)
    input_paths.extend([loss_data_path, actuation_data_path])

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
    throughput = actuation.groupby("path_index").throughput_mvarh.mean().reindex(x)
    saturation = actuation.groupby("path_index").saturation_percent.mean().reindex(x)
    ax.plot(x, throughput, color=ORANGE, marker="o", linewidth=1.4)
    ax.set_xticks(x, labels, rotation=45)
    ax.set_xlabel("Capacity point")
    ax.set_ylabel("Reactive throughput (MVArh/day)", color=ORANGE)
    ax.tick_params(axis="y", labelcolor=ORANGE)
    ax2 = ax.twinx()
    ax2.plot(x, saturation, color=BLUE, marker="s", linewidth=1.2)
    ax2.set_ylabel("Action saturation (%)", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    panel_label(ax, "c", "Actuation tradeoffs")
    tidy_axis(ax)
    for point, text_value, colour in (
        (int(rejected["path_index"]), "Rejected", RED),
        (int(selected["path_index"]), "Selected", GREEN),
    ):
        ax.axvline(point, color=colour, linestyle="--", linewidth=0.9)
        ax.text(
            point + 0.08,
            0.96 if text_value == "Selected" else 0.84,
            text_value,
            transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
            color=CHARCOAL,
            fontsize=6.5,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": colour},
        )

    efficiency = claim["confirmed_boundary_operational_comparison"]
    if claim["permitted_claims"].get(
        "no_greater_capacity_and_lower_loss_vs_pilot_droop"
    ):
        fig.text(
            0.50,
            -0.02,
            "Confirmed VMOD vs pilot droop: "
            f"{efficiency['loss_change_percent_of_droop']:.2f}% line-loss change; "
            f"95% CI [{efficiency['loss_ci95_low_mwh']:.4f}, "
            f"{efficiency['loss_ci95_high_mwh']:.4f}] MWh/day",
            ha="center",
            fontsize=7.3,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.30", "fc": "white", "ec": "#9CA3AF"},
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
        figsize=(7.20, 3.35),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.45, 1.05, 1.0]},
    )

    ax = axes[0]
    y = np.arange(len(labels))
    low = 100 * levels.plot_event_rate_ci95_low.to_numpy(dtype=float)
    high = 100 * levels.plot_event_rate_ci95_high.to_numpy(dtype=float)
    xerr = np.vstack([rates - low, high - rates])
    normalizer = mpl.colors.Normalize(vmin=0, vmax=max(1.0, float(rates.max())))
    colours = plt.get_cmap("YlOrRd")(normalizer(rates))
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
    ax.scatter(rates, y, c=colours, edgecolor=CHARCOAL, linewidth=0.5, s=30, zorder=2)
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
    panel_label(ax, "a", "Shift degradation")
    tidy_axis(ax, "x")

    ax = axes[1]
    systems = [("IEEE 33-bus", main_boundary), ("IEEE 69-bus", env69_boundary)]
    for y, (name, boundary) in enumerate(systems):
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
            ax.plot(
                [0, int(env69_path.path_index.max())],
                [y, y],
                color="#D1D5DB",
                linewidth=1.0,
                zorder=0,
            )
            ax.text(
                0.25,
                y,
                "No development point passed (15 tested)",
                color=RED,
                va="center",
                ha="left",
                fontsize=5.3,
                bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": RED},
            )
            continue
        ax.plot(
            [rejected["path_index"], selected["path_index"]],
            [y, y],
            color=CHARCOAL,
            linestyle="--",
            linewidth=1.0,
        )
        ax.scatter(
            rejected["path_index"],
            y,
            marker="X",
            s=65,
            color=RED,
            edgecolor=CHARCOAL,
            linewidth=0.7,
        )
        ax.scatter(
            selected["path_index"],
            y,
            marker="D",
            s=60,
            color=GREEN,
            edgecolor=CHARCOAL,
            linewidth=0.7,
        )
    ax.set_yticks(range(len(systems)), [item[0] for item in systems])
    maximum_index = max(
        [
            int(row["path_index"])
            for _, boundary in systems
            for row in boundary.get("points", [])
        ]
        + [8, int(env69_path.path_index.max())]
    )
    tick_indices = sorted(set([0, 3, 6, 9, 12, maximum_index]))
    ax.set_xticks(tick_indices, [f"P{i:02d}" for i in tick_indices])
    ax.set_xlim(-0.5, maximum_index + 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Capacity-path index")
    panel_label(ax, "b", "Adapted boundary")
    tidy_axis(ax, "x")
    ax.legend(
        handles=[
            mpl.lines.Line2D([], [], marker="X", linestyle="", color=RED, label="Rejected"),
            mpl.lines.Line2D([], [], marker="D", linestyle="", color=GREEN, label="Selected"),
        ],
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#9CA3AF",
        fontsize=6.4,
    )

    ax = axes[2]
    columns = ["actor_inference_ms", "ac_execution_ms", "total_ms"]
    labels = ["Actor", "AC plant", "Total"]
    box = ax.boxplot(
        [runtime_steps[column].to_numpy() for column in columns],
        labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )
    for patch, colour in zip(box["boxes"], [ORANGE, CHARCOAL, BLUE]):
        patch.set_facecolor(mpl.colors.to_rgba(colour, 0.35))
        patch.set_edgecolor(colour)
    ax.set_yscale("log")
    ax.set_ylabel("Latency per step (ms, log scale)")
    panel_label(ax, "c", "Online latency")
    tidy_axis(ax)
    ax.text(
        0.04,
        0.96,
        f"Total p95: {runtime['total_online_path_ms']['p95']:.3f} ms\n"
        f"Maximum: {runtime['total_online_path_ms']['maximum']:.3f} ms\n"
        "Control interval: 900,000 ms",
        transform=ax.transAxes,
        va="top",
        fontsize=6.7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#9CA3AF"},
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        default="protocol",
        help="Comma-separated: protocol,margin,boundary,tradeoff,robustness,all",
    )
    args = parser.parse_args()
    requested = {item.strip().lower() for item in args.figures.split(",") if item.strip()}
    if "all" in requested:
        requested = {"protocol", "margin", "boundary", "tradeoff", "robustness"}
    unknown = requested - {"protocol", "margin", "boundary", "tradeoff", "robustness"}
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
    if "robustness" in requested:
        robustness_portability_runtime_figure()


if __name__ == "__main__":
    main()
