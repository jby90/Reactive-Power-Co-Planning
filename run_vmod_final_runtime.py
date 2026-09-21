"""Run an uncontended final VMOD latency benchmark after heavy experiments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from vmod_current_study import (
    EXTERNAL_PROFILES_33,
    EXTERNAL_PROTOCOL_33,
    env69_run_dir,
    main_run_dir,
    read_selected_margin,
)


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
PROTOCOL = EXTERNAL_PROTOCOL_33


def main() -> None:
    margin = read_selected_margin()
    main_run = main_run_dir(margin)
    env69_run = env69_run_dir(margin)
    required = (
        main_run / "confirmation_boundary_summary.json",
        main_run / "conditioning_ablation" / "summary.json",
        env69_run / "confirmation_boundary_summary.json",
        main_run / "operational_shift" / "study_summary.json",
        main_run / "power_balance_audit" / "summary.json",
        main_run / "final_baselines" / "summary.json",
        main_run / "conventional_capacity_boundaries" / "summary.json",
    )
    while not all(path.exists() for path in required):
        time.sleep(60)
    boundary = json.loads(required[0].read_text(encoding="utf-8"))
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
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    subprocess.run(
        [
            str(PYTHON),
            "-u",
            "benchmark_vmod_runtime.py",
            "--theta",
            theta,
            "--days_metadata",
            str(PROTOCOL / "selection_days.csv"),
            "--profile_dir",
            str(EXTERNAL_PROFILES_33),
            "--checkpoint",
            str(main_run / "policies" / "policy_init_seed42.pth"),
            "--svc_absorption_ratio",
            "1.0",
            "--out_dir",
            str(main_run / "runtime_benchmark"),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    subprocess.run(
        [
            str(PYTHON),
            "-u",
            "benchmark_vmod_scalability.py",
            "--envs",
            "33,69,118",
            "--steps",
            "192",
            "--out_dir",
            str(main_run / "runtime_scalability"),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
