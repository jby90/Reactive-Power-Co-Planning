"""Apply the frozen 17-day selection gate to a trained VMOD capacity path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
SEEDS = (42, 43, 44, 45, 46)
MIN_TEACHER_COVERAGE_FRACTION = 0.8


def teacher_coverage_eligible(attempted_days: int, converged_days: int) -> bool:
    return bool(
        attempted_days > 0
        and converged_days / attempted_days >= MIN_TEACHER_COVERAGE_FRACTION
    )


def upper_path_is_monotone(frame: pd.DataFrame, selected_index: int | None) -> bool:
    """Require every tested point above the first pass to pass as well."""
    if selected_index is None:
        return False
    upper = frame.loc[frame.path_index >= selected_index]
    return bool(not upper.empty and upper.passed_selection_gate.all())


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path_run_dir", type=Path, required=True)
    parser.add_argument("--protocol_dir", type=Path, required=True)
    parser.add_argument("--profile_dir", type=Path, required=True)
    parser.add_argument("--env", type=int, default=33, choices=[33, 69, 118])
    parser.add_argument("--cap_buses", default="20,8")
    parser.add_argument("--svc_absorption_ratio", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    run_dir = args.path_run_dir.resolve()
    protocol = args.protocol_dir.resolve()
    profiles = args.profile_dir.resolve()
    policies = run_dir / "policies"
    path = pd.read_csv(protocol / "capacity_path.csv").sort_values("path_index")
    dataset_manifest = json.loads(
        (run_dir / "dataset" / "manifest.json").read_text(encoding="utf-8")
    )
    coverage = {
        int(row["candidate_id"]): row
        for row in dataset_manifest["candidate_coverage"]
    }
    rows = []
    for point in path.itertuples(index=False):
        out_dir = run_dir / "selection" / point.capacity_label
        if not (out_dir / "summary.json").exists():
            run(
                [
                    str(PYTHON),
                    "-u",
                    "evaluate_opf_initialisation_baseline.py",
                    "--theta",
                    f"{point.pv_s_scale},{point.svc_q_scale},{point.cap_total_mvar}",
                    "--days_metadata",
                    str(protocol / "selection_days.csv"),
                    "--profile_dir",
                    str(profiles),
                    "--initialisation_dir",
                    str(policies),
                    "--seeds",
                    ",".join(map(str, SEEDS)),
                    "--workers",
                    str(args.workers),
                    "--svc_absorption_ratio",
                    str(args.svc_absorption_ratio),
                    "--env",
                    str(args.env),
                    "--cap_buses",
                    args.cap_buses,
                    "--out_dir",
                    str(out_dir),
                ]
            )
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        seed_rows = summary["seed_summaries"]
        support = coverage[int(point.path_index)]
        attempted_days = int(support["attempted_days"])
        converged_days = int(support["converged_days"])
        rows.append(
            {
                **point._asdict(),
                "all_seeds_zero_events": bool(summary["all_seeds_zero_events"]),
                "all_seeds_zero_pf_failures": all(
                    int(row["pf_fail_days"]) == 0 for row in seed_rows
                ),
                "teacher_attempted_days": attempted_days,
                "complete_teacher_trajectory_days": converged_days,
                "teacher_coverage_fraction": converged_days / attempted_days,
                "adequate_teacher_trajectory_support": teacher_coverage_eligible(
                    attempted_days, converged_days
                ),
                "total_event_days": int(summary["total_event_days"]),
                "maximum_seed_event_days": int(summary["maximum_seed_event_days"]),
                "mean_seed_loss_mwh": float(summary["mean_seed_loss_mwh"]),
                "worst_vmin": min(
                    float(row["worst_minimum_voltage_pu"]) for row in seed_rows
                ),
                "worst_vmax": max(
                    float(row["worst_maximum_voltage_pu"]) for row in seed_rows
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["passed_selection_gate"] = (
        frame.all_seeds_zero_events
        & frame.all_seeds_zero_pf_failures
        & frame.adequate_teacher_trajectory_support
    )
    frame["rejection_mechanism"] = "passed"
    frame.loc[
        ~frame.adequate_teacher_trajectory_support,
        "rejection_mechanism",
    ] = "inadequate_teacher_trajectory_support"
    frame.loc[
        frame.adequate_teacher_trajectory_support & ~frame.all_seeds_zero_pf_failures,
        "rejection_mechanism",
    ] = "power_flow_failure"
    frame.loc[
        frame.adequate_teacher_trajectory_support
        & frame.all_seeds_zero_pf_failures
        & ~frame.all_seeds_zero_events,
        "rejection_mechanism",
    ] = "voltage_event"
    eligible = frame.loc[frame.passed_selection_gate].sort_values("path_index")
    selected_index = None if eligible.empty else int(eligible.iloc[0].path_index)
    rejected_index = None
    if selected_index is not None and selected_index > int(frame.path_index.min()):
        previous = frame.loc[frame.path_index == selected_index - 1].iloc[0]
        if not bool(previous.passed_selection_gate):
            rejected_index = int(previous.path_index)
    adjacent_transition_identified = bool(
        selected_index is not None and rejected_index is not None
    )
    all_higher_points_pass = upper_path_is_monotone(frame, selected_index)
    frame["selected_for_confirmation"] = frame.path_index.eq(selected_index)
    frame["adjacent_rejected_for_confirmation"] = frame.path_index.eq(rejected_index)
    frame.to_csv(run_dir / "selection_path_summary.csv", index=False)
    summary = {
        "selection_days_per_seed": 17,
        "training_seeds": list(SEEDS),
        "selected_path_index": selected_index,
        "adjacent_rejected_path_index": rejected_index,
        "adjacent_transition_identified": adjacent_transition_identified,
        "all_higher_points_pass": all_higher_points_pass,
        "boundary_identified": bool(
            adjacent_transition_identified and all_higher_points_pass
        ),
        "selection_rule": (
            "Lowest nested physical-capacity point with zero event-days and zero "
            "power-flow failures for every one of five training seeds and complete "
            "teacher trajectories on at least 80% of fitting days; every higher "
            "tested path point must also pass before the transition is called a "
            "capacity boundary."
        ),
        "minimum_teacher_coverage_fraction": MIN_TEACHER_COVERAGE_FRACTION,
        "confirmation_data_used": False,
        "points": frame.to_dict("records"),
    }
    (run_dir / "selection_path_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
