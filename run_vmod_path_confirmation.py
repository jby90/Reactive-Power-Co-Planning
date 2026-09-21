"""Confirm the selected VMOD capacity point and its adjacent rejected point."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from vmod_statistics import clopper_pearson_upper


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
SEEDS = (42, 43, 44, 45, 46)


def common_confirmation_days(point_results: list[dict]) -> int:
    counts = {
        int(row["evaluated_days"])
        for point in point_results
        for row in point["seed_summaries"]
    }
    if len(counts) != 1:
        raise ValueError(f"Inconsistent confirmation-day counts: {sorted(counts)}")
    return counts.pop()


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
    selection = json.loads(
        (run_dir / "selection_path_summary.json").read_text(encoding="utf-8")
    )
    selected_index = selection["selected_path_index"]
    rejected_index = selection["adjacent_rejected_path_index"]
    if not selection.get("boundary_identified", False):
        raise RuntimeError(
            "Selection did not identify a monotone rejected/passed capacity boundary"
        )
    path = pd.read_csv(protocol / "capacity_path.csv").set_index("path_index")
    point_results = []
    for role, path_index in (
        ("adjacent_rejected", int(rejected_index)),
        ("selected", int(selected_index)),
    ):
        point = path.loc[path_index]
        out_dir = run_dir / "confirmation" / str(point.capacity_label)
        if not (out_dir / "summary.json").exists():
            run(
                [
                    str(PYTHON),
                    "-u",
                    "evaluate_opf_initialisation_baseline.py",
                    "--theta",
                    f"{point.pv_s_scale},{point.svc_q_scale},{point.cap_total_mvar}",
                    "--days_metadata",
                    str(protocol / "confirmation_days.csv"),
                    "--profile_dir",
                    str(profiles),
                    "--initialisation_dir",
                    str(run_dir / "policies"),
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
        result = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        per_seed = []
        for row in result["seed_summaries"]:
            events = int(row["event_days"])
            trials = int(row["evaluated_days"])
            per_seed.append(
                {
                    **row,
                    "observed_event_rate": events / trials,
                    "one_sided_clopper_pearson_upper_95": clopper_pearson_upper(
                        events, trials
                    ),
                }
            )
        point_results.append(
            {
                "role": role,
                "path_index": path_index,
                "capacity_label": str(point.capacity_label),
                "pv_s_scale": float(point.pv_s_scale),
                "svc_q_scale": float(point.svc_q_scale),
                "cap_total_mvar": float(point.cap_total_mvar),
                "pv_inverter_nameplate_mva": float(
                    point.pv_inverter_nameplate_mva
                ),
                "svc_nameplate_mvar": float(point.svc_nameplate_mvar),
                "all_seeds_zero_events": bool(result["all_seeds_zero_events"]),
                "all_seeds_zero_pf_failures": all(
                    int(row["pf_fail_days"]) == 0 for row in per_seed
                ),
                "total_event_days": int(result["total_event_days"]),
                "mean_seed_loss_mwh": float(result["mean_seed_loss_mwh"]),
                "seed_summaries": per_seed,
            }
        )

    rejected = next(row for row in point_results if row["role"] == "adjacent_rejected")
    selected = next(row for row in point_results if row["role"] == "selected")
    confirmation_days = common_confirmation_days(point_results)
    selected_pass = bool(
        selected["all_seeds_zero_events"]
        and selected["all_seeds_zero_pf_failures"]
        and all(
            float(row["one_sided_clopper_pearson_upper_95"]) < 0.01
            for row in selected["seed_summaries"]
        )
    )
    rejected_fails = bool(
        (not rejected["all_seeds_zero_events"])
        or (not rejected["all_seeds_zero_pf_failures"])
    )
    summary = {
        "boundary_confirmed": bool(selected_pass and rejected_fails),
        "selected_point_passed": selected_pass,
        "adjacent_lower_point_failed": rejected_fails,
        "confirmation_days_per_seed": confirmation_days,
        "inference_unit": "day, evaluated separately for each training seed",
        "dependence_note": (
            "The five policies share the same operating days, so seed-day records "
            "are not pooled as independent binomial trials."
        ),
        "points": point_results,
    }
    (run_dir / "confirmation_boundary_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
