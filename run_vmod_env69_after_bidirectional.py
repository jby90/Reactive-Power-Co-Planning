"""Run the IEEE 69-bus VMOD portability path after the 33-bus main path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from vmod_current_study import (
    DEVELOPMENT_PROFILES_69,
    DEVELOPMENT_PROTOCOL_69,
    EXTERNAL_PROFILES_69,
    EXTERNAL_PROTOCOL_69,
    env69_run_dir,
    main_run_dir,
    read_selected_margin,
)


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)


def unconfirmed_boundary_payload(selection: dict) -> dict:
    """Represent a failed portability selection without fabricating confirmation."""
    return {
        "boundary_confirmed": False,
        "confirmation_opened": False,
        "selected_point_passed": False,
        "adjacent_lower_point_failed": False,
        "reason": "external selection did not identify a monotone adjacent rejected/passed boundary",
        "selection_summary": selection,
        "points": [],
    }


def development_range_has_passing_point(calibration: pd.DataFrame) -> bool:
    """Require a development-feasible point before opening external data."""
    required = {"all_seeds_zero_events", "evaluated_seed_days"}
    missing = sorted(required - set(calibration.columns))
    if missing:
        raise ValueError(f"Calibration summary is missing columns: {missing}")
    passed = calibration.all_seeds_zero_events
    if passed.dtype != bool:
        passed = passed.astype(str).str.lower().map({"true": True, "false": False})
    return bool(passed.fillna(False).any())


def main() -> None:
    margin = read_selected_margin()
    main_boundary = main_run_dir(margin) / "confirmation_boundary_summary.json"
    while not main_boundary.exists():
        time.sleep(60)
    if not json.loads(main_boundary.read_text(encoding="utf-8")).get(
        "boundary_confirmed", False
    ):
        raise RuntimeError("The 33-bus external boundary did not pass confirmation")

    out = env69_run_dir(margin)
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
            "run_vmod_path_candidate.py",
            "--out_dir",
            str(out),
            "--protocol_dir",
            str(DEVELOPMENT_PROTOCOL_69),
            "--profile_dir",
            str(DEVELOPMENT_PROFILES_69),
            "--margin",
            str(margin),
            "--svc_absorption_ratio",
            "1.0",
            "--env",
            "69",
            "--cap_buses",
            "",
            "--build_teacher",
            "--teacher_workers",
            "8",
            "--dataset_workers",
            "8",
            "--allow_missing_validation_days",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    calibration_path = out / "calibration_path_summary.csv"
    calibration = pd.read_csv(calibration_path)
    if not development_range_has_passing_point(calibration):
        payload = {
            "boundary_confirmed": False,
            "confirmation_opened": False,
            "selected_point_passed": False,
            "adjacent_lower_point_failed": False,
            "reason": (
                "no point in the frozen extended path passed development "
                "calibration; external selection was not opened"
            ),
            "development_calibration_summary": str(calibration_path),
            "points": [],
        }
        (out / "confirmation_boundary_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, indent=2))
        return
    common = [
        "--path_run_dir",
        str(out),
        "--protocol_dir",
        str(EXTERNAL_PROTOCOL_69),
        "--profile_dir",
        str(EXTERNAL_PROFILES_69),
        "--env",
        "69",
        "--cap_buses",
        "",
        "--svc_absorption_ratio",
        "1.0",
    ]
    subprocess.run(
        [str(PYTHON), "-u", "run_vmod_path_selection.py", *common],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    selection = json.loads(
        (out / "selection_path_summary.json").read_text(encoding="utf-8")
    )
    if not selection.get("boundary_identified", False):
        payload = unconfirmed_boundary_payload(selection)
        (out / "confirmation_boundary_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, indent=2))
        return
    subprocess.run(
        [str(PYTHON), "-u", "run_vmod_path_confirmation.py", *common],
        cwd=ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
