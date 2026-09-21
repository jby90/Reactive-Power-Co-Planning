"""Authoritative paths and identifiers for the redesigned VMOD study."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DEVELOPMENT_MARGIN_ROOT = (
    ROOT / "runs" / "VMOD_BIDIRECTIONAL_MARGIN_STUDY_20260920"
)
PATH_MARGIN_SUMMARY = (
    ROOT / "runs" / "VMOD_PATH_MARGIN_CALIBRATION_20260921" / "summary.json"
)

DEVELOPMENT_PROTOCOL_33 = ROOT / "runs" / "VMOD_PROTOCOL_MAIN_20260920"
DEVELOPMENT_PROTOCOL_69 = (
    ROOT / "runs" / "VMOD_PROTOCOL_MAIN_ENV69_EXTENDED_20260921"
)
DEVELOPMENT_PROFILES_33 = ROOT / "data" / "vmod" / "profiles33"
DEVELOPMENT_PROFILES_69 = ROOT / "data" / "vmod" / "profiles69"

EXTERNAL_PROTOCOL_33 = ROOT / "runs" / "VMOD_PROTOCOL_EXTERNAL2018_20260921"
EXTERNAL_PROTOCOL_69 = (
    ROOT / "runs" / "VMOD_PROTOCOL_EXTERNAL2018_ENV69_EXTENDED_20260921"
)
EXTERNAL_PROFILES_33 = ROOT / "data" / "vmod" / "profiles33_external2018"
EXTERNAL_PROFILES_69 = ROOT / "data" / "vmod" / "profiles69_external2018"
FINAL_PROTOCOL_69 = (
    ROOT / "runs" / "VMOD_PROTOCOL_EXTERNAL2019_ENV69_FINAL_20260921"
)
FINAL_PROFILES_69 = ROOT / "data" / "vmod" / "profiles69_external2019"

FINAL_ROOT = ROOT / "runs" / "VMOD_FINAL_STUDY_20260921"


def margin_label(margin: float) -> str:
    return str(float(margin)).replace(".", "p")


def read_selected_margin() -> float:
    payload = json.loads(PATH_MARGIN_SUMMARY.read_text(encoding="utf-8"))
    margin = payload.get("selected_margin_pu")
    if margin is None:
        raise RuntimeError("No path-calibrated VMOD margin is available")
    if payload.get("development_selection_data_used") is not False:
        raise RuntimeError("Margin selection unexpectedly used development selection data")
    if payload.get("external_validation_data_used") is not False:
        raise RuntimeError("Margin selection unexpectedly used external validation data")
    return float(margin)


def main_run_dir(margin: float | None = None) -> Path:
    selected = read_selected_margin() if margin is None else float(margin)
    return ROOT / "runs" / f"VMOD_EXTERNAL2018_MARGIN_{margin_label(selected)}_20260921"


def env69_run_dir(margin: float | None = None) -> Path:
    selected = read_selected_margin() if margin is None else float(margin)
    if selected != 0.003:
        raise RuntimeError("The audited 69-bus adaptation uses the frozen 0.003-pu margin")
    return ROOT / "runs" / "VMOD_ENV69_ADAPTED_FINAL_20260921"


def protocol_day_count(protocol_dir: Path, split: str) -> int:
    path = protocol_dir / f"{split}_days.csv"
    with path.open("r", encoding="utf-8") as stream:
        return max(sum(1 for _ in stream) - 1, 0)


def profile_windows_are_disjoint(
    development_profiles: Path, external_profiles: Path
) -> bool:
    development = json.loads(
        (development_profiles / "metadata.json").read_text(encoding="utf-8")
    )
    external = json.loads(
        (external_profiles / "metadata.json").read_text(encoding="utf-8")
    )
    development_end = datetime.fromisoformat(
        development["window_end_utc_inclusive"]
    )
    external_start = datetime.fromisoformat(external["window_start_utc"])
    return bool(
        development_end < external_start
        and development["source_file_sha256"] == external["source_file_sha256"]
    )
