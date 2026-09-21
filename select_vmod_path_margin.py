"""Freeze a teacher margin from development calibration data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MIN_TEACHER_COVERAGE = 0.8
MIN_CALIBRATION_VOLTAGE_PU = 0.951


def candidate_path_gate(
    path: pd.DataFrame,
    coverage: pd.DataFrame,
    minimum_voltage_pu: float = MIN_CALIBRATION_VOLTAGE_PU,
) -> dict:
    frame = path.merge(
        coverage[["candidate_id", "attempted_days", "converged_days"]],
        left_on="path_index",
        right_on="candidate_id",
        validate="one_to_one",
    ).copy()
    frame["teacher_coverage_fraction"] = (
        frame.converged_days / frame.attempted_days
    )
    frame["adequate_teacher_support"] = (
        frame.teacher_coverage_fraction >= MIN_TEACHER_COVERAGE
    )
    frame["passes_calibration"] = (
        frame.adequate_teacher_support
        & frame.all_seeds_zero_events.astype(bool)
        & frame.all_seeds_zero_pf_failures.astype(bool)
        & (frame.worst_vmin >= minimum_voltage_pu)
    )
    passing = frame.loc[frame.passes_calibration].sort_values("path_index")
    first_index = None if passing.empty else int(passing.iloc[0].path_index)
    all_higher_pass = bool(
        first_index is not None
        and frame.loc[frame.path_index >= first_index, "passes_calibration"].all()
    )
    return {
        "first_passing_path_index": first_index,
        "all_higher_points_pass": all_higher_pass,
        "passed_path_gate": bool(first_index is not None and all_higher_pass),
        "minimum_calibration_voltage_pu": minimum_voltage_pu,
        "points": frame.to_dict("records"),
    }


def load_candidate(candidate: Path, margin: float) -> dict:
    path_csv = candidate / "calibration_path_summary.csv"
    manifest_path = candidate / "dataset" / "manifest.json"
    if not path_csv.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete path calibration at {candidate}")
    path = pd.read_csv(path_csv)
    pf_values = []
    for row in path.itertuples(index=False):
        summary = json.loads(
            (candidate / "calibration" / row.capacity_label / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        pf_values.append(
            all(int(item["pf_fail_days"]) == 0 for item in summary["seed_summaries"])
        )
    path["all_seeds_zero_pf_failures"] = pf_values
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coverage = pd.DataFrame(manifest["candidate_coverage"])
    return {
        "margin_pu": margin,
        "candidate_dir": str(candidate),
        **candidate_path_gate(path, coverage),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin_root", type=Path, required=True)
    parser.add_argument(
        "--zero_candidate_dir",
        type=Path,
        help="Optional exact frozen zero-margin run containing path calibration.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidates = []
    for margin in (0.0, 0.001, 0.003, 0.005):
        label = f"margin_{int(round(margin * 1000)):03d}"
        candidate = (
            args.zero_candidate_dir
            if margin == 0.0 and args.zero_candidate_dir is not None
            else args.margin_root / label
        )
        candidates.append(load_candidate(candidate, margin))
    passing = [row for row in candidates if row["passed_path_gate"]]
    selected = None if not passing else min(passing, key=lambda row: row["margin_pu"])
    payload = {
        "schema_version": 1,
        "selection_rule": (
            "Smallest teacher margin whose five actors have zero calibration "
            "event-days and PF failures, minimum voltage at least 0.951 pu, and "
            "no higher-capacity reversal after the first passing path point."
        ),
        "development_selection_data_used": False,
        "external_validation_data_used": False,
        "selected_margin_pu": None if selected is None else selected["margin_pu"],
        "selected_candidate_dir": None if selected is None else selected["candidate_dir"],
        "candidates": candidates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if selected is None:
        raise RuntimeError("No teacher margin passed the path-wide calibration gate")


if __name__ == "__main__":
    main()
