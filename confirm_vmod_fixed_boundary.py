"""Confirm a frozen adjacent VMOD boundary on one untouched profile window."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from vmod_statistics import clopper_pearson_upper


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
SEEDS = (42, 43, 44, 45, 46)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_evaluation(command: list[str]) -> None:
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies_dir", type=Path, required=True)
    parser.add_argument("--protocol_dir", type=Path, required=True)
    parser.add_argument("--profile_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--selected_index", type=int, required=True)
    parser.add_argument("--rejected_index", type=int, required=True)
    parser.add_argument("--env", type=int, default=69)
    parser.add_argument("--cap_buses", default="")
    parser.add_argument("--svc_absorption_ratio", type=float, default=1.0)
    parser.add_argument("--workers_per_point", type=int, default=5)
    args = parser.parse_args()

    protocol = args.protocol_dir.resolve()
    profiles = args.profile_dir.resolve()
    policies = args.policies_dir.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    days_path = protocol / "confirmation_days.csv"
    path_table = pd.read_csv(protocol / "capacity_path.csv").set_index("path_index")
    if args.selected_index != args.rejected_index + 1:
        raise ValueError("The rejected and selected points must be adjacent")

    frozen = {
        "role": "untouched-window fixed-boundary confirmation",
        "selected_path_index": args.selected_index,
        "adjacent_rejected_path_index": args.rejected_index,
        "training_seeds": list(SEEDS),
        "confirmation_days": int(len(pd.read_csv(days_path))),
        "confirmation_days_sha256": file_sha256(days_path),
        "capacity_path_sha256": file_sha256(protocol / "capacity_path.csv"),
        "profile_metadata_sha256": file_sha256(profiles / "metadata.json"),
        "policy_checkpoint_sha256": {
            str(seed): file_sha256(policies / f"policy_init_seed{seed}.pth")
            for seed in SEEDS
        },
    }
    (out / "fixed_boundary_protocol.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )

    jobs = []
    point_specs = []
    for role, index in (
        ("adjacent_rejected", args.rejected_index),
        ("selected", args.selected_index),
    ):
        point = path_table.loc[index]
        point_out = out / role / str(point.capacity_label)
        point_specs.append((role, index, point, point_out))
        jobs.append(
            [
                str(PYTHON),
                "-u",
                "evaluate_opf_initialisation_baseline.py",
                "--theta",
                f"{point.pv_s_scale},{point.svc_q_scale},{point.cap_total_mvar}",
                "--days_metadata",
                str(days_path),
                "--profile_dir",
                str(profiles),
                "--initialisation_dir",
                str(policies),
                "--seeds",
                ",".join(map(str, SEEDS)),
                "--workers",
                str(args.workers_per_point),
                "--svc_absorption_ratio",
                str(args.svc_absorption_ratio),
                "--env",
                str(args.env),
                "--cap_buses",
                args.cap_buses,
                "--out_dir",
                str(point_out),
            ]
        )
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run_evaluation, jobs))

    points = []
    for role, index, point, point_out in point_specs:
        result = json.loads((point_out / "summary.json").read_text(encoding="utf-8"))
        seed_summaries = []
        for row in result["seed_summaries"]:
            events = int(row["event_days"])
            trials = int(row["evaluated_days"])
            seed_summaries.append(
                {
                    **row,
                    "observed_event_rate": events / trials,
                    "one_sided_clopper_pearson_upper_95": clopper_pearson_upper(
                        events, trials
                    ),
                }
            )
        points.append(
            {
                "role": role,
                "path_index": int(index),
                "capacity_label": str(point.capacity_label),
                "pv_s_scale": float(point.pv_s_scale),
                "svc_q_scale": float(point.svc_q_scale),
                "cap_total_mvar": float(point.cap_total_mvar),
                "total_event_days": int(result["total_event_days"]),
                "all_seeds_zero_events": bool(result["all_seeds_zero_events"]),
                "all_seeds_zero_pf_failures": bool(
                    result["all_seeds_zero_pf_failures"]
                ),
                "mean_seed_loss_mwh": float(result["mean_seed_loss_mwh"]),
                "seed_summaries": seed_summaries,
            }
        )

    rejected = next(point for point in points if point["role"] == "adjacent_rejected")
    selected = next(point for point in points if point["role"] == "selected")
    selected_passed = bool(
        selected["all_seeds_zero_events"]
        and selected["all_seeds_zero_pf_failures"]
        and all(
            row["one_sided_clopper_pearson_upper_95"] < 0.01
            for row in selected["seed_summaries"]
        )
    )
    rejected_failed = bool(
        not rejected["all_seeds_zero_events"]
        or not rejected["all_seeds_zero_pf_failures"]
    )
    summary = {
        **frozen,
        "confirmation_days_per_seed": frozen["confirmation_days"],
        "boundary_confirmed": bool(selected_passed and rejected_failed),
        "selected_point_passed": selected_passed,
        "adjacent_lower_point_failed": rejected_failed,
        "inference_unit": "day, evaluated separately for each training seed",
        "points": points,
    }
    (out / "confirmation_boundary_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
