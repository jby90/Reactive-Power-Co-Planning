"""Map tuned-droop and no-control capacity boundaries on the VMOD path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from vmod_statistics import clopper_pearson_upper
from vmod_current_study import (
    DEVELOPMENT_PROFILES_33,
    DEVELOPMENT_PROTOCOL_33,
    EXTERNAL_PROFILES_33,
    EXTERNAL_PROTOCOL_33,
    main_run_dir,
    protocol_day_count,
    read_selected_margin,
)


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
PROTOCOL = EXTERNAL_PROTOCOL_33
PROFILES = EXTERNAL_PROFILES_33
COARSE_SCALES = (0.10, 0.25, 0.50, 0.75, 1.00)
REFINEMENT_SCALES = (0.04, 0.06, 0.08, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22)
PILOT_SCALES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.00)
PILOT_TUNED_MAX_PATH_INDEX = 3
TUNING_OUTER_WORKERS = 1
TUNING_COLUMNS = (
    "action_scale",
    "full_injection",
    "deadband_low",
    "deadband_high",
    "full_absorption",
)


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


def theta_string(point) -> str:
    return f"{point.pv_s_scale},{point.svc_q_scale},{point.cap_total_mvar}"


def confirmation_point_passed(events: int, trials: int, pf_failures: int) -> bool:
    return bool(
        events == 0
        and pf_failures == 0
        and clopper_pearson_upper(events, trials) < 0.01
    )


def all_higher_points_pass(
    frame: pd.DataFrame, method: str, selected_index: int | None
) -> bool:
    if selected_index is None:
        return False
    upper = frame.loc[
        (frame.method == method) & (frame.path_index >= selected_index)
    ].sort_values("path_index")
    return bool(not upper.empty and upper.passed_selection_gate.all())


def merge_droop_tuning_tables(
    coarse: pd.DataFrame, refinement: pd.DataFrame
) -> pd.DataFrame:
    """Combine two development-only grids and retain each configuration once."""
    required = (
        *TUNING_COLUMNS,
        "pf_fail_days",
        "event_days",
        "mean_daily_line_loss_mwh",
    )
    missing = [
        column
        for column in required
        if column not in coarse or column not in refinement
    ]
    if missing:
        raise ValueError(f"Droop tuning table is missing columns: {missing}")
    combined = pd.concat((coarse, refinement), ignore_index=True)
    combined = combined.drop_duplicates(subset=list(TUNING_COLUMNS), keep="first")
    return combined.sort_values(
        ["pf_fail_days", "event_days", "mean_daily_line_loss_mwh"],
        kind="stable",
    ).reset_index(drop=True)


def ensure_final_tuning(point, out: Path) -> dict:
    """Run development-only coarse/refined tuning and freeze the merged optimum."""
    point_dir = out / point.capacity_label
    coarse = point_dir / "droop_tuning"
    if not (coarse / "frozen_droop.json").exists():
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
                theta_string(point),
                "--svc_absorption_ratio",
                "1.0",
                "--workers",
                "6",
                "--out_dir",
                str(coarse),
            ]
        )
    refinement = point_dir / "droop_tuning_refinement"
    if not (refinement / "frozen_droop.json").exists():
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
                theta_string(point),
                "--svc_absorption_ratio",
                "1.0",
                "--workers",
                "6",
                "--scales",
                ",".join(map(str, REFINEMENT_SCALES)),
                "--out_dir",
                str(refinement),
            ]
        )
    final = point_dir / "droop_tuning_final"
    table = merge_droop_tuning_tables(
        pd.read_csv(coarse / "training_grid.csv"),
        pd.read_csv(refinement / "training_grid.csv"),
    )
    expected = 3 * (len(COARSE_SCALES) + len(REFINEMENT_SCALES))
    if len(table) != expected:
        raise RuntimeError(
            f"Expected {expected} unique droop configurations at "
            f"{point.capacity_label}, got {len(table)}"
        )
    best = table.iloc[0].to_dict()
    payload = {
        "selection_rule": "lexicographic PF failures, event days, then mean line loss",
        "development_only_tuning": True,
        "theta": [
            float(point.pv_s_scale),
            float(point.svc_q_scale),
            float(point.cap_total_mvar),
        ],
        "environment": 33,
        "svc_absorption_ratio": 1.0,
        "training_day_count": protocol_day_count(DEVELOPMENT_PROTOCOL_33, "fitting"),
        "coarse_action_scales": list(COARSE_SCALES),
        "refinement_action_scales": list(REFINEMENT_SCALES),
        "configurations": int(len(table)),
        "best": best,
    }
    final.mkdir(parents=True, exist_ok=True)
    table.to_csv(final / "training_grid.csv", index=False)
    (final / "frozen_droop.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return best


def ensure_pilot_tuning(point, out: Path) -> dict:
    """Tune the stronger feeder-wide pilot droop on development days only."""
    point_dir = out / point.capacity_label / "pilot_droop_tuning"
    if not (point_dir / "frozen_droop.json").exists():
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
                theta_string(point),
                "--svc_absorption_ratio",
                "1.0",
                "--workers",
                "6",
                "--method",
                "pilot_droop",
                "--scales",
                ",".join(map(str, PILOT_SCALES)),
                "--out_dir",
                str(point_dir),
            ]
        )
    payload = json.loads((point_dir / "frozen_droop.json").read_text(encoding="utf-8"))
    expected = 4 * len(PILOT_SCALES)
    if payload.get("method") != "pilot_droop" or payload.get("configurations") != expected:
        raise RuntimeError(
            f"Expected {expected} pilot-droop configurations at "
            f"{point.capacity_label}"
        )
    return payload["best"]


def main() -> None:
    margin = read_selected_margin()
    main_run = main_run_dir(margin)
    boundary_file = main_run / "confirmation_boundary_summary.json"
    while not boundary_file.exists():
        time.sleep(60)
    if not json.loads(boundary_file.read_text(encoding="utf-8"))["boundary_confirmed"]:
        raise RuntimeError("The VMOD boundary did not pass confirmation")

    out = main_run / "conventional_capacity_boundaries"
    path = pd.read_csv(PROTOCOL / "capacity_path.csv").sort_values("path_index")
    rows = []
    frozen_configs: dict[int, dict] = {}
    points = list(path.itertuples(index=False))
    # Each tuner owns a six-process pool; serial outer dispatch avoids spawning
    # several independent pandapower pools at once on memory-constrained hosts.
    with ThreadPoolExecutor(max_workers=TUNING_OUTER_WORKERS) as executor:
        futures = {
            int(point.path_index): executor.submit(ensure_final_tuning, point, out)
            for point in points
        }
        for index, future in futures.items():
            frozen_configs[index] = future.result()
    pilot_configs: dict[int, dict] = {}
    for point in points:
        index = int(point.path_index)
        if index <= PILOT_TUNED_MAX_PATH_INDEX:
            pilot_configs[index] = ensure_pilot_tuning(point, out)
        else:
            # Freeze the first development-feasible pilot curve at P03 and use
            # it unchanged above the candidate boundary. This tests higher-path
            # monotonicity without using external outcomes to retune the law.
            pilot_configs[index] = dict(
                pilot_configs[PILOT_TUNED_MAX_PATH_INDEX]
            )
    for point in points:
        point_dir = out / point.capacity_label
        best = frozen_configs[int(point.path_index)]
        selection = point_dir / "selection_refined"
        if not (selection / "summary.json").exists():
            run(
                [
                    "evaluate_raw_non_rl_baselines.py",
                    "--theta",
                    theta_string(point),
                    "--days_metadata",
                    str(PROTOCOL / "selection_days.csv"),
                    "--load_profile",
                    str(PROFILES / "load_15min_366d.npy"),
                    "--generation_profile",
                    str(PROFILES / "pv_15min_366d.npy"),
                    "--methods",
                    "droop,no_control",
                    "--action_scale",
                    str(best["action_scale"]),
                    "--full_injection",
                    str(best["full_injection"]),
                    "--deadband_low",
                    str(best["deadband_low"]),
                    "--deadband_high",
                    str(best["deadband_high"]),
                    "--full_absorption",
                    str(best["full_absorption"]),
                    "--svc_absorption_ratio",
                    "1.0",
                    "--out_dir",
                    str(selection),
                ]
            )
        summary = pd.read_csv(selection / "summary.csv")
        for result in summary.itertuples(index=False):
            rows.append(
                {
                    **point._asdict(),
                    "method": str(result.method),
                    "selection_event_days": int(result.event_days),
                    "selection_pf_fail_days": int(result.pf_fail_days),
                    "selection_mean_daily_loss_mwh": float(
                        result.mean_daily_line_loss_mwh
                    ),
                    "passed_selection_gate": bool(
                        result.event_days == 0 and result.pf_fail_days == 0
                    ),
                }
            )
        pilot_best = pilot_configs[int(point.path_index)]
        pilot_selection = point_dir / "selection_pilot"
        if not (pilot_selection / "summary.json").exists():
            run(
                [
                    "evaluate_raw_non_rl_baselines.py",
                    "--theta",
                    theta_string(point),
                    "--days_metadata",
                    str(PROTOCOL / "selection_days.csv"),
                    "--load_profile",
                    str(PROFILES / "load_15min_366d.npy"),
                    "--generation_profile",
                    str(PROFILES / "pv_15min_366d.npy"),
                    "--methods",
                    "pilot_droop",
                    "--action_scale",
                    str(pilot_best["action_scale"]),
                    "--full_injection",
                    str(pilot_best["full_injection"]),
                    "--deadband_low",
                    str(pilot_best["deadband_low"]),
                    "--deadband_high",
                    str(pilot_best["deadband_high"]),
                    "--full_absorption",
                    str(pilot_best["full_absorption"]),
                    "--svc_absorption_ratio",
                    "1.0",
                    "--out_dir",
                    str(pilot_selection),
                ]
            )
        pilot_result = pd.read_csv(pilot_selection / "summary.csv").iloc[0]
        rows.append(
            {
                **point._asdict(),
                "method": "pilot_droop",
                "selection_event_days": int(pilot_result.event_days),
                "selection_pf_fail_days": int(pilot_result.pf_fail_days),
                "selection_mean_daily_loss_mwh": float(
                    pilot_result.mean_daily_line_loss_mwh
                ),
                "passed_selection_gate": bool(
                    pilot_result.event_days == 0
                    and pilot_result.pf_fail_days == 0
                ),
            }
        )

    frame = pd.DataFrame(rows)
    selected_indices: dict[str, int | None] = {}
    rejected_indices: dict[str, int | None] = {}
    upper_path_monotone: dict[str, bool] = {}
    for method in ("pilot_droop", "droop", "no_control"):
        eligible = frame.query(
            "method == @method and passed_selection_gate"
        ).sort_values("path_index")
        selected_indices[method] = (
            None if eligible.empty else int(eligible.iloc[0].path_index)
        )
        selected_index = selected_indices[method]
        rejected_index = None
        if selected_index is not None and selected_index > int(path.path_index.min()):
            previous = frame.loc[
                (frame.method == method)
                & (frame.path_index == selected_index - 1)
            ].iloc[0]
            if not bool(previous.passed_selection_gate):
                rejected_index = selected_index - 1
        rejected_indices[method] = rejected_index
        upper_path_monotone[method] = all_higher_points_pass(
            frame, method, selected_index
        )
    frame["selected_for_confirmation"] = False
    frame["adjacent_rejected_for_confirmation"] = False
    for method, index in selected_indices.items():
        if index is not None and upper_path_monotone[method]:
            frame.loc[
                (frame.method == method) & (frame.path_index == index),
                "selected_for_confirmation",
            ] = True
        rejected_index = rejected_indices[method]
        if rejected_index is not None and upper_path_monotone[method]:
            frame.loc[
                (frame.method == method) & (frame.path_index == rejected_index),
                "adjacent_rejected_for_confirmation",
            ] = True
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "selection_path_summary.csv", index=False)

    confirmation = []
    for method, selected_index in selected_indices.items():
        rejected_index = rejected_indices[method]
        if (
            selected_index is None
            or rejected_index is None
            or not upper_path_monotone[method]
        ):
            confirmation.append(
                {
                    "method": method,
                    "selected_path_index": selected_index,
                    "adjacent_rejected_path_index": rejected_index,
                    "all_higher_points_pass": upper_path_monotone[method],
                    "boundary_confirmed": False,
                    "passed": False,
                }
            )
            continue
        point_results = {}
        for role, index in (
            ("selected", selected_index),
            ("adjacent_rejected", rejected_index),
        ):
            point = path.loc[path.path_index == index].iloc[0]
            best = (
                pilot_configs[index]
                if method == "pilot_droop"
                else frozen_configs[index]
            )
            confirm_dir = (
                out
                / str(point.capacity_label)
                / "confirmation_refined"
                / method
            )
            if not (confirm_dir / "summary.json").exists():
                arguments = [
                    "evaluate_raw_non_rl_baselines.py",
                    "--theta",
                    f"{point.pv_s_scale},{point.svc_q_scale},{point.cap_total_mvar}",
                    "--days_metadata",
                    str(PROTOCOL / "confirmation_days.csv"),
                    "--load_profile",
                    str(PROFILES / "load_15min_366d.npy"),
                    "--generation_profile",
                    str(PROFILES / "pv_15min_366d.npy"),
                    "--methods",
                    method,
                    "--svc_absorption_ratio",
                    "1.0",
                    "--out_dir",
                    str(confirm_dir),
                ]
                if method in {"droop", "pilot_droop"}:
                    arguments.extend(
                        [
                            "--action_scale",
                            str(best["action_scale"]),
                            "--full_injection",
                            str(best["full_injection"]),
                            "--deadband_low",
                            str(best["deadband_low"]),
                            "--deadband_high",
                            str(best["deadband_high"]),
                            "--full_absorption",
                            str(best["full_absorption"]),
                        ]
                    )
                run(arguments)
            result = pd.read_csv(confirm_dir / "summary.csv").iloc[0]
            events = int(result.event_days)
            trials = int(result.evaluated_days)
            pf_failures = int(result.pf_fail_days)
            point_results[role] = {
                "path_index": int(index),
                "capacity_label": str(point.capacity_label),
                "pv_inverter_nameplate_mva": float(
                    point.pv_inverter_nameplate_mva
                ),
                "svc_nameplate_mvar": float(point.svc_nameplate_mvar),
                "event_days": events,
                "evaluated_days": trials,
                "observed_event_rate": events / trials,
                "one_sided_clopper_pearson_upper_95": clopper_pearson_upper(
                    events, trials
                ),
                "pf_fail_days": pf_failures,
                "mean_daily_line_loss_mwh": float(
                    result.mean_daily_line_loss_mwh
                ),
                "mean_daily_absolute_reactive_throughput_mvarh": float(
                    result.mean_daily_absolute_reactive_throughput_mvarh
                ),
                "passed": confirmation_point_passed(
                    events, trials, pf_failures
                ),
            }
        selected_result = point_results["selected"]
        rejected_result = point_results["adjacent_rejected"]
        adjacent_lower_failed = bool(
            rejected_result["event_days"] > 0
            or rejected_result["pf_fail_days"] > 0
        )
        boundary_confirmed = bool(
            selected_result["passed"] and adjacent_lower_failed
        )
        confirmation.append(
            {
                "method": method,
                "selected_path_index": selected_index,
                "adjacent_rejected_path_index": rejected_index,
                "selected_point_passed": bool(selected_result["passed"]),
                "adjacent_lower_point_failed": adjacent_lower_failed,
                "all_higher_points_pass": upper_path_monotone[method],
                "boundary_confirmed": boundary_confirmed,
                "passed": boundary_confirmed,
                "points": point_results,
            }
        )
    summary = {
        "droop_tuning": {
            "development_only": True,
            "coarse_action_scales": list(COARSE_SCALES),
            "refinement_action_scales": list(REFINEMENT_SCALES),
            "breakpoint_curves": 3,
            "unique_configurations_per_path_point": 3
            * (len(COARSE_SCALES) + len(REFINEMENT_SCALES)),
        },
        "pilot_droop_tuning": {
            "development_only": True,
            "action_scales": list(PILOT_SCALES),
            "breakpoint_curves": 4,
            "unique_configurations_per_tuned_path_point": 4 * len(PILOT_SCALES),
            "independently_tuned_through_path_index": PILOT_TUNED_MAX_PATH_INDEX,
            "higher_path_rule": (
                "Freeze the P03 development-selected curve and apply it unchanged "
                "at every higher path point."
            ),
        },
        "selection_rule": (
            "Tune conventional droop only on 40 development fitting days; choose "
            "the lowest point with zero events and PF failures on 17 selection days."
        ),
        "selected_path_indices": selected_indices,
        "adjacent_rejected_path_indices": rejected_indices,
        "all_higher_points_pass": upper_path_monotone,
        "confirmation_rule": (
            "A conventional-control boundary is confirmed only when its selected "
            f"point passes the {protocol_day_count(PROTOCOL, 'confirmation')}-day "
            "finite-sample gate, the adjacent lower point exhibits at least one "
            "voltage event or power-flow failure, and every higher tested path "
            "point passed external selection."
        ),
        "confirmation": confirmation,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
