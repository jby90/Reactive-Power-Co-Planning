"""Build matched conventional and optimization baselines at the final VMOD point."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from vmod_statistics import mean_t_interval
from vmod_current_study import (
    DEVELOPMENT_PROFILES_33,
    DEVELOPMENT_PROTOCOL_33,
    EXTERNAL_PROFILES_33,
    EXTERNAL_PROTOCOL_33,
    DEVELOPMENT_MARGIN_ROOT,
    main_run_dir,
    protocol_day_count,
    read_selected_margin,
)


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
PROTOCOL = EXTERNAL_PROTOCOL_33
PROFILES = EXTERNAL_PROFILES_33


def environment() -> dict[str, str]:
    values = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        values[name] = "1"
    return values


def run(arguments: list[str]) -> None:
    subprocess.run([str(PYTHON), "-u", *arguments], cwd=ROOT, env=environment(), check=True)


def paired_seed_interval(
    proposed: pd.DataFrame, baseline: pd.DataFrame, baseline_loss: str
) -> dict:
    values = []
    for seed, group in proposed.groupby("seed", sort=True):
        paired = group.merge(baseline, on="day", validate="one_to_one")
        if len(paired) != len(group) or len(paired) != len(baseline):
            raise ValueError(
                f"Seed {int(seed)} does not have an exactly matched baseline day set"
            )
        values.append(
            {
                "seed": int(seed),
                "paired_days": int(len(paired)),
                "vmod_minus_baseline_loss_mwh": float(
                    (paired.line_loss_mwh - paired[baseline_loss]).mean()
                ),
            }
        )
    mean, low, high = mean_t_interval(
        pd.DataFrame(values).vmod_minus_baseline_loss_mwh.to_numpy()
    )
    return {
        "mean_mwh": mean,
        "seed_level_t_ci95_low_mwh": low,
        "seed_level_t_ci95_high_mwh": high,
        "seed_values": values,
    }


def paired_matched_seed_interval(proposed: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    """Pair policies by training seed and operating day."""
    values = []
    for seed, group in proposed.groupby("seed", sort=True):
        seed_baseline = baseline.query("seed == @seed")
        paired = group.merge(
            seed_baseline,
            on="day",
            suffixes=("_vmod", "_baseline"),
            validate="one_to_one",
        )
        if len(paired) != len(group) or len(paired) != len(seed_baseline):
            raise ValueError(
                f"Seed {int(seed)} does not have an exactly matched baseline day set"
            )
        values.append(
            {
                "seed": int(seed),
                "paired_days": int(len(paired)),
                "vmod_event_days": int(paired.event_vmod.sum()),
                "baseline_event_days": int(paired.event_baseline.sum()),
                "vmod_minus_baseline_loss_mwh": float(
                    (
                        paired.line_loss_mwh_vmod
                        - paired.line_loss_mwh_baseline
                    ).mean()
                ),
            }
        )
    mean, low, high = mean_t_interval(
        pd.DataFrame(values).vmod_minus_baseline_loss_mwh.to_numpy()
    )
    return {
        "mean_mwh": mean,
        "seed_level_t_ci95_low_mwh": low,
        "seed_level_t_ci95_high_mwh": high,
        "seed_values": values,
    }


def operational_summary(frame: pd.DataFrame) -> dict:
    result = {
        "event_days": int(frame.event.sum()),
        "pf_fail_days": int(frame.pf_fail.sum()),
        "mean_daily_loss_mwh": float(frame.line_loss_mwh.mean()),
        "worst_minimum_voltage_pu": float(frame.minimum_voltage_pu.min()),
        "worst_maximum_voltage_pu": float(frame.maximum_voltage_pu.max()),
    }
    for column in (
        "mean_absolute_normalized_action",
        "normalized_action_saturation_fraction",
        "absolute_reactive_throughput_mvarh",
        "maximum_absolute_reactive_command_mvar",
    ):
        if column in frame:
            reducer = "max" if column.startswith("maximum_") else "mean"
            result[column] = float(getattr(frame[column], reducer)())
    return result


def build_path_selection_jobs(
    days: pd.DataFrame, capacity_path: pd.DataFrame
) -> pd.DataFrame:
    """Return the full day-by-capacity product without changing either order."""
    if days.empty or capacity_path.empty:
        raise ValueError("Selection days and capacity path must both be non-empty")
    if "path_index" not in capacity_path:
        raise ValueError("Capacity path is missing path_index")
    path = capacity_path.copy()
    if "candidate_id" not in path:
        path["candidate_id"] = path["path_index"].astype(int)
    return days.assign(_join=1).merge(
        path.assign(_join=1), on="_join", validate="many_to_many"
    ).drop(columns="_join")


def main() -> None:
    margin = read_selected_margin()
    main_run = main_run_dir(margin)
    boundary_path = main_run / "confirmation_boundary_summary.json"
    while not boundary_path.exists():
        time.sleep(60)
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    if not boundary["boundary_confirmed"]:
        raise RuntimeError("The main capacity boundary did not pass confirmation")
    selected = next(
        point for point in boundary["points"] if point["role"] == "selected"
    )
    theta = ",".join(
        map(
            str,
            (
                selected["pv_s_scale"],
                selected["svc_q_scale"],
                selected["cap_total_mvar"],
            ),
        )
    )
    out = main_run / "final_baselines"
    shared_droop_tuning = (
        main_run
        / "conventional_capacity_boundaries"
        / str(selected["capacity_label"])
        / "droop_tuning_final"
    )
    droop_tuning = (
        shared_droop_tuning
        if (shared_droop_tuning / "frozen_droop.json").exists()
        else out / "droop_tuning"
    )
    if not (droop_tuning / "frozen_droop.json").exists():
        run(
            [
                "tune_raw_droop_baseline.py",
                "--training_days",
                str(DEVELOPMENT_PROTOCOL_33 / "fitting_days.csv"),
                "--load_profile",
                str(DEVELOPMENT_PROFILES_33 / "load_15min_366d.npy"),
                "--generation_profile",
                str(DEVELOPMENT_PROFILES_33 / "pv_15min_366d.npy"),
                "--theta",
                theta,
                "--svc_absorption_ratio",
                "1.0",
                "--workers",
                "6",
                "--out_dir",
                str(droop_tuning),
            ]
        )
    frozen = json.loads((droop_tuning / "frozen_droop.json").read_text(encoding="utf-8"))[
        "best"
    ]
    conventional = out / "conventional_selection"
    if not (conventional / "summary.json").exists():
        run(
            [
                "evaluate_raw_non_rl_baselines.py",
                "--theta",
                theta,
                "--days_metadata",
                str(PROTOCOL / "selection_days.csv"),
                "--load_profile",
                str(PROFILES / "load_15min_366d.npy"),
                "--generation_profile",
                str(PROFILES / "pv_15min_366d.npy"),
                "--methods",
                "droop,no_control",
                "--action_scale",
                str(frozen["action_scale"]),
                "--full_injection",
                str(frozen["full_injection"]),
                "--deadband_low",
                str(frozen["deadband_low"]),
                "--deadband_high",
                str(frozen["deadband_high"]),
                "--full_absorption",
                str(frozen["full_absorption"]),
                "--svc_absorption_ratio",
                "1.0",
                "--out_dir",
                str(conventional),
            ]
        )

    conventional_confirmation = out / "conventional_confirmation"
    if not (conventional_confirmation / "summary.json").exists():
        run(
            [
                "evaluate_raw_non_rl_baselines.py",
                "--theta",
                theta,
                "--days_metadata",
                str(PROTOCOL / "confirmation_days.csv"),
                "--load_profile",
                str(PROFILES / "load_15min_366d.npy"),
                "--generation_profile",
                str(PROFILES / "pv_15min_366d.npy"),
                "--methods",
                "droop,no_control",
                "--action_scale",
                str(frozen["action_scale"]),
                "--full_injection",
                str(frozen["full_injection"]),
                "--deadband_low",
                str(frozen["deadband_low"]),
                "--deadband_high",
                str(frozen["deadband_high"]),
                "--full_absorption",
                str(frozen["full_absorption"]),
                "--svc_absorption_ratio",
                "1.0",
                "--out_dir",
                str(conventional_confirmation),
            ]
        )

    zero_margin_root = (
        DEVELOPMENT_MARGIN_ROOT
        / "margin_000"
        / "policies"
    )
    zero_margin_outputs = {}
    for split in ("selection", "confirmation"):
        zero_out = out / f"zero_margin_{split}"
        zero_margin_outputs[split] = zero_out
        if not (zero_out / "summary.json").exists():
            run(
                [
                    "evaluate_opf_initialisation_baseline.py",
                    "--theta",
                    theta,
                    "--days_metadata",
                    str(PROTOCOL / f"{split}_days.csv"),
                    "--profile_dir",
                    str(PROFILES),
                    "--initialisation_dir",
                    str(zero_margin_root),
                    "--seeds",
                    "42,43,44,45,46",
                    "--workers",
                    "5",
                    "--svc_absorption_ratio",
                    "1.0",
                    "--out_dir",
                    str(zero_out),
                ]
            )

    # Build the AC-OPF comparator over the complete capacity path on exactly
    # the same selection days.  A selected-point-only run cannot distinguish
    # controller regret from a genuine capacity/loss envelope.
    opf = out / "ac_opf_path_selection"
    if not (opf / "summary.json").exists():
        days = pd.read_csv(PROTOCOL / "selection_days.csv")
        capacity_path = pd.read_csv(PROTOCOL / "capacity_path.csv")
        jobs = build_path_selection_jobs(days, capacity_path)
        out.mkdir(parents=True, exist_ok=True)
        jobs_path = out / "ac_opf_path_selection_jobs.csv"
        jobs.to_csv(jobs_path, index=False)
        run(
            [
                "ac_opf_nested_envelope.py",
                "--jobs_csv",
                str(jobs_path),
                "--out_dir",
                str(opf),
                "--workers",
                "4",
                "--lower_voltage",
                "0.95",
                "--upper_voltage",
                "1.05",
                "--load_profile",
                str(PROFILES / "load_15min_366d.npy"),
                "--generation_profile",
                str(PROFILES / "pv_15min_366d.npy"),
                "--svc_absorption_ratio",
                "1.0",
                "--no_direct_fallback",
            ]
        )

    proposed = pd.read_csv(
        main_run / "selection" / str(selected["capacity_label"]) / "daily.csv"
    )
    proposed_confirmation = pd.read_csv(
        main_run
        / "confirmation"
        / str(selected["capacity_label"])
        / "daily.csv"
    )
    conventional_daily = pd.read_csv(conventional / "daily.csv")
    conventional_confirmation_daily = pd.read_csv(
        conventional_confirmation / "daily.csv"
    )
    pilot_root = (
        main_run
        / "conventional_capacity_boundaries"
        / str(selected["capacity_label"])
    )
    pilot_selection_daily = pd.read_csv(
        pilot_root / "selection_pilot" / "daily.csv"
    )
    pilot_confirmation_daily = pd.read_csv(
        pilot_root
        / "confirmation_refined"
        / "pilot_droop"
        / "daily.csv"
    )
    conventional_daily = pd.concat(
        (conventional_daily, pilot_selection_daily), ignore_index=True
    )
    conventional_confirmation_daily = pd.concat(
        (conventional_confirmation_daily, pilot_confirmation_daily),
        ignore_index=True,
    )
    zero_margin_selection = pd.read_csv(
        zero_margin_outputs["selection"] / "daily.csv"
    )
    zero_margin_confirmation = pd.read_csv(
        zero_margin_outputs["confirmation"] / "daily.csv"
    )
    opf_path_daily = pd.read_csv(opf / "episodes.csv")
    opf_daily = opf_path_daily.loc[
        opf_path_daily.candidate_id == int(selected["path_index"])
    ].copy()
    opf_daily = opf_daily.rename(
        columns={"day_index": "day", "daily_loss_mwh": "opf_loss_mwh"}
    )[["day", "risk", "pf_fail", "opf_loss_mwh"]]
    rows = []
    for method, group in conventional_daily.groupby("method", sort=True):
        rows.append(
            {
                "baseline": method,
                "split": "selection",
                **operational_summary(group),
                "paired_loss": paired_seed_interval(
                    proposed,
                    group[["day", "line_loss_mwh"]].rename(
                        columns={"line_loss_mwh": "baseline_loss_mwh"}
                    ),
                    "baseline_loss_mwh",
                ),
            }
        )
    rows.append(
        {
            "baseline": "centralized_local_ac_opf",
            "split": "selection",
            "event_days": int(opf_daily.risk.sum()),
            "pf_fail_days": int(opf_daily.pf_fail.sum()),
            "mean_daily_loss_mwh": float(opf_daily.opf_loss_mwh.mean()),
            "direct_fallback_steps": int(
                opf_path_daily.loc[
                    opf_path_daily.candidate_id == int(selected["path_index"]),
                    "direct_fallback_steps",
                ].sum()
            ),
            "paired_loss": paired_seed_interval(proposed, opf_daily, "opf_loss_mwh"),
        }
    )
    rows.append(
        {
            "baseline": "zero_margin_opf_imitation",
            "split": "selection",
            **operational_summary(zero_margin_selection),
            "paired_loss": paired_matched_seed_interval(
                proposed, zero_margin_selection
            ),
        }
    )
    for method, group in conventional_confirmation_daily.groupby(
        "method", sort=True
    ):
        rows.append(
            {
                "baseline": method,
                "split": "confirmation",
                **operational_summary(group),
                "paired_loss": paired_seed_interval(
                    proposed_confirmation,
                    group[["day", "line_loss_mwh"]].rename(
                        columns={"line_loss_mwh": "baseline_loss_mwh"}
                    ),
                    "baseline_loss_mwh",
                ),
            }
        )
    rows.append(
        {
            "baseline": "zero_margin_opf_imitation",
            "split": "confirmation",
            **operational_summary(zero_margin_confirmation),
            "paired_loss": paired_matched_seed_interval(
                proposed_confirmation, zero_margin_confirmation
            ),
        }
    )
    summary = {
        "capacity_label": selected["capacity_label"],
        "theta": theta,
        "vmod_event_days_across_five_seeds": int(proposed.event.sum()),
        "vmod_mean_seed_loss_mwh": float(
            proposed.groupby("seed").line_loss_mwh.mean().mean()
        ),
        "vmod_selection": operational_summary(proposed),
        "vmod_confirmation": operational_summary(proposed_confirmation),
        "droop_tuning_days": protocol_day_count(
            DEVELOPMENT_PROTOCOL_33, "fitting"
        ),
        "selection_days": protocol_day_count(PROTOCOL, "selection"),
        "confirmation_days": protocol_day_count(PROTOCOL, "confirmation"),
        "ac_opf_path_selection_dir": str(opf),
        "ac_opf_path_selection_points": int(
            opf_path_daily.candidate_id.nunique()
        ),
        "comparisons": rows,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
