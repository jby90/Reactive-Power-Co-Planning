"""Run the final VMOD AC nodal-balance audit after boundary confirmation."""

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
    main_run_dir,
    read_selected_margin,
)


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
PROTOCOL = EXTERNAL_PROTOCOL_33


def main() -> None:
    margin = read_selected_margin()
    run_dir = main_run_dir(margin)
    boundary_path = run_dir / "confirmation_boundary_summary.json"
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
            "audit_vmod_power_balance.py",
            "--theta",
            theta,
            "--days_metadata",
            str(PROTOCOL / "selection_days.csv"),
            "--profile_dir",
            str(EXTERNAL_PROFILES_33),
            "--initialisation_dir",
            str(run_dir / "policies"),
            "--seeds",
            "42,43,44,45,46",
            "--svc_absorption_ratio",
            "1.0",
            "--out_dir",
            str(run_dir / "power_balance_audit"),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
