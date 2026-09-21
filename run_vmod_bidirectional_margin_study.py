"""Calibrate the VMOD teacher margin under the main bidirectional-SVC model."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
STUDY = ROOT / "runs" / "VMOD_BIDIRECTIONAL_MARGIN_STUDY_20260920"
CANDIDATES = (
    (0.005, "margin_005"),
    (0.003, "margin_003"),
    (0.001, "margin_001"),
    (0.0, "margin_000"),
)


def candidate_passed(row: dict) -> bool:
    return bool(
        row["all_seeds_zero_events"]
        and row["all_seeds_zero_pf_failures"]
        and row["adequate_calibration_teacher_support"]
    )


def load_candidate_result(label: str) -> dict:
    path = STUDY / label / "calibration_result.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    calibration_label = row["calibration_reference_label"]
    evaluation_path = STUDY / label / f"calibration_{calibration_label}" / "summary.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    row["all_seeds_zero_pf_failures"] = evaluation.get(
        "all_seeds_zero_pf_failures",
        all(
            int(seed_row["pf_fail_days"]) == 0
            for seed_row in evaluation["seed_summaries"]
        ),
    )
    row["total_pf_fail_days"] = evaluation.get(
        "total_pf_fail_days",
        sum(int(seed_row["pf_fail_days"]) for seed_row in evaluation["seed_summaries"]),
    )
    manifest = json.loads(
        (STUDY / label / "dataset" / "manifest.json").read_text(encoding="utf-8")
    )
    support = next(
        item
        for item in manifest["candidate_coverage"]
        if item["capacity_label"] == calibration_label
    )
    attempted = int(support["attempted_days"])
    converged = int(support["converged_days"])
    fraction = 0.0 if attempted == 0 else converged / attempted
    row["calibration_reference_teacher_attempted_days"] = attempted
    row["calibration_reference_teacher_converged_days"] = converged
    row["calibration_reference_teacher_coverage_fraction"] = fraction
    row["adequate_calibration_teacher_support"] = fraction >= 0.8
    row["passed_calibration_gate"] = candidate_passed(row)
    path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


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


def main() -> None:
    for margin, label in CANDIDATES:
        out = STUDY / label
        if (out / "calibration_result.json").exists():
            continue
        subprocess.run(
            [
                str(PYTHON),
                "-u",
                "run_vmod_margin_candidate.py",
                "--margin",
                str(margin),
                "--out_dir",
                str(out),
                "--teacher_workers",
                "4",
                "--dataset_workers",
                "1",
                "--svc_absorption_ratio",
                "1.0",
                "--jobs_csv",
                str(
                    ROOT
                    / "runs"
                    / "VMOD_PROTOCOL_MAIN_20260920"
                    / "training_path_jobs.csv"
                ),
                "--calibration_days",
                str(
                    ROOT
                    / "runs"
                    / "VMOD_PROTOCOL_MAIN_20260920"
                    / "calibration_days.csv"
                ),
                "--student_validation_days",
                str(
                    ROOT
                    / "runs"
                    / "VMOD_PROTOCOL_MAIN_20260920"
                    / "student_validation_days.csv"
                ),
                "--no_direct_fallback",
            ],
            cwd=ROOT,
            env=environment(),
            check=True,
        )
    rows = [load_candidate_result(label) for _, label in CANDIDATES]
    frame = pd.DataFrame(rows).sort_values("margin_pu", ignore_index=True)
    eligible = frame.loc[frame.passed_calibration_gate]
    selected_margin = None if eligible.empty else float(eligible.iloc[0].margin_pu)
    frame["selected"] = frame.margin_pu.eq(selected_margin)
    STUDY.mkdir(parents=True, exist_ok=True)
    frame.to_csv(STUDY / "margin_calibration_summary.csv", index=False)
    summary = {
        "summary_schema_version": 2,
        "device_model": "bidirectional SVC with symmetric absorption and injection",
        "svc_absorption_ratio": 1.0,
        "selection_rule": (
            "Smallest teacher margin with zero event-days and zero PF failures "
            "for all five seeds on the frozen ten-day calibration split."
        ),
        "selected_margin_pu": selected_margin,
        "selection_or_confirmation_data_used": False,
        "candidates": frame.to_dict("records"),
    }
    (STUDY / "margin_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
