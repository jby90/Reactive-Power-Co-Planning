"""Freeze the path-calibrated VMOD candidate and validate it on 2018 data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_vmod_bidirectional_after_margin import freeze_calibrated_candidate
from vmod_current_study import (
    DEVELOPMENT_MARGIN_ROOT,
    EXTERNAL_PROFILES_33,
    EXTERNAL_PROTOCOL_33,
    PATH_MARGIN_SUMMARY,
    main_run_dir,
)


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
MARGIN_ROOT = DEVELOPMENT_MARGIN_ROOT
ZERO_RUN = ROOT / "runs" / "VMOD_PATH_BIDIRECTIONAL_MARGIN_0p0_20260920"
SELECTION_SUMMARY = PATH_MARGIN_SUMMARY
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


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, env=environment(), check=True)


def wait_for_path_calibrations() -> None:
    required = [ZERO_RUN / "calibration_path_summary.csv"] + [
        MARGIN_ROOT / label / "calibration_path_summary.csv"
        for label in ("margin_001", "margin_003", "margin_005")
    ]
    while not all(path.is_file() for path in required):
        time.sleep(20)


def main() -> None:
    wait_for_path_calibrations()
    run(
        [
            str(PYTHON),
            "select_vmod_path_margin.py",
            "--margin_root",
            str(MARGIN_ROOT),
            "--zero_candidate_dir",
            str(ZERO_RUN),
            "--out",
            str(SELECTION_SUMMARY),
        ]
    )
    calibration = json.loads(SELECTION_SUMMARY.read_text(encoding="utf-8"))
    margin = calibration["selected_margin_pu"]
    if margin is None:
        raise RuntimeError("No path-calibrated teacher margin is available")
    source = Path(calibration["selected_candidate_dir"])
    out = main_run_dir(margin)
    frozen = freeze_calibrated_candidate(source, out)
    external_manifest = {
        "role": "post-development external validation",
        "selected_margin_pu": margin,
        "path_calibration_summary": str(SELECTION_SUMMARY),
        "source_candidate": str(source),
        "frozen_artifact_manifest": str(out / "frozen_margin_candidate.json"),
        "frozen_artifact_count": frozen["frozen_artifact_count"],
        "profile_metadata": str(PROFILES / "metadata.json"),
        "protocol_metadata": str(PROTOCOL / "external_validation_protocol.json"),
        "development_selection_results_used": False,
    }
    (out / "external_validation_manifest.json").write_text(
        json.dumps(external_manifest, indent=2), encoding="utf-8"
    )
    common = [
        "--path_run_dir",
        str(out),
        "--protocol_dir",
        str(PROTOCOL),
        "--profile_dir",
        str(PROFILES),
        "--env",
        "33",
        "--cap_buses",
        "20,8",
        "--svc_absorption_ratio",
        "1.0",
    ]
    run([str(PYTHON), "-u", "run_vmod_path_selection.py", *common])
    selection = json.loads(
        (out / "selection_path_summary.json").read_text(encoding="utf-8")
    )
    if not selection.get("boundary_identified", False):
        raise RuntimeError(
            "External 2018 selection did not identify a monotone capacity boundary"
        )
    run([str(PYTHON), "-u", "run_vmod_path_confirmation.py", *common])


if __name__ == "__main__":
    main()
