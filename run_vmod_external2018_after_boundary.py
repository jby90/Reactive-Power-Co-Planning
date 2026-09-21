"""Complete all claim-gated VMOD evidence after external confirmation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from vmod_current_study import main_run_dir, read_selected_margin


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)


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


def run(script: str, *arguments: str) -> None:
    subprocess.run(
        [str(PYTHON), "-u", script, *arguments],
        cwd=ROOT,
        env=environment(),
        check=True,
    )


def wait_for_confirmed_main_boundary() -> Path:
    run_dir = main_run_dir(read_selected_margin())
    boundary_path = run_dir / "confirmation_boundary_summary.json"
    while not boundary_path.is_file():
        time.sleep(30)
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    if not boundary.get("boundary_confirmed", False):
        raise RuntimeError(
            "The 33-bus external boundary failed; downstream superiority evidence "
            "must not be generated from an unconfirmed point"
        )
    return run_dir


def main() -> None:
    wait_for_confirmed_main_boundary()

    # Conventional boundaries come first because their tuned curves are reused by
    # the selected-point matched comparisons.
    run("run_vmod_conventional_capacity_boundaries.py")
    run("run_vmod_final_baselines.py")
    run("run_vmod_conditioning_ablation.py")
    run("run_vmod_power_balance_after_boundary.py")
    run("run_vmod_env69_after_bidirectional.py")
    run("run_vmod_operational_shift_study.py")
    run("run_vmod_final_runtime.py")
    run("summarize_vmod_final_study.py")
    run("make_vmod_figures.py", "--figures", "all")


if __name__ == "__main__":
    main()
