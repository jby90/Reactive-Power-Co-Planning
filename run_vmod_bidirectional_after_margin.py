"""Run the bidirectional-SVC VMOD capacity path after margin calibration freezes."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from vmod_protocol import wait_for_audited_margin_summary


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
MARGIN_SUMMARY = (
    ROOT
    / "runs"
    / "VMOD_BIDIRECTIONAL_MARGIN_STUDY_20260920"
    / "margin_calibration_summary.json"
)
SEEDS = (42, 43, 44, 45, 46)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_calibrated_candidate(source: Path, destination: Path) -> dict:
    """Copy the exact calibrated teacher, dataset, and actors into the main run."""
    required = [
        source / "teacher" / "summary.json",
        source / "teacher" / "steps.csv",
        source / "teacher" / "episodes.csv",
        source / "dataset" / "opf_imitation_dataset.npz",
        source / "dataset" / "manifest.json",
        *[
            source / "policies" / f"policy_init_seed{seed}.pth"
            for seed in SEEDS
        ],
        *[
            source / "policies" / f"summary_seed{seed}.json"
            for seed in SEEDS
        ],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Calibrated candidate is incomplete: {missing}")

    copied = []
    source_roots = ("teacher", "dataset", "policies")
    for directory in source_roots:
        source_dir = source / directory
        for source_file in sorted(source_dir.glob("*")):
            if not source_file.is_file():
                continue
            destination_file = destination / directory / source_file.name
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            source_hash = sha256(source_file)
            if destination_file.exists():
                if sha256(destination_file) != source_hash:
                    raise RuntimeError(
                        f"Frozen artifact mismatch at {destination_file}"
                    )
            else:
                shutil.copy2(source_file, destination_file)
            destination_hash = sha256(destination_file)
            if destination_hash != source_hash:
                raise RuntimeError(f"Copy verification failed for {source_file}")
            copied.append(
                {
                    "source": str(source_file),
                    "destination": str(destination_file),
                    "sha256": source_hash,
                    "bytes": int(source_file.stat().st_size),
                }
            )

    manifest = {
        "source_candidate": str(source),
        "destination_run": str(destination),
        "frozen_artifact_count": len(copied),
        "files": copied,
        "note": (
            "The exact actors used to select the voltage margin are reused for "
            "capacity-path calibration, selection, and confirmation."
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "frozen_margin_candidate.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    payload = wait_for_audited_margin_summary(MARGIN_SUMMARY)
    margin = payload.get("selected_margin_pu")
    if margin is None:
        raise RuntimeError("No voltage margin passed the frozen calibration gate")
    label = str(margin).replace(".", "p")
    out = ROOT / "runs" / f"VMOD_PATH_BIDIRECTIONAL_MARGIN_{label}_20260920"
    candidate_label = f"margin_{int(round(float(margin) * 1000)):03d}"
    source = (
        ROOT
        / "runs"
        / "VMOD_BIDIRECTIONAL_MARGIN_STUDY_20260920"
        / candidate_label
    )
    freeze_calibrated_candidate(source, out)
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
            str(ROOT / "runs" / "VMOD_PROTOCOL_MAIN_20260920"),
            "--margin",
            str(margin),
            "--svc_absorption_ratio",
            "1.0",
            "--dataset_workers",
            "6",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    common = [
        "--path_run_dir",
        str(out),
        "--protocol_dir",
        str(ROOT / "runs" / "VMOD_PROTOCOL_MAIN_20260920"),
        "--profile_dir",
        str(ROOT / "data" / "vmod" / "profiles33"),
        "--env",
        "33",
        "--cap_buses",
        "20,8",
        "--svc_absorption_ratio",
        "1.0",
    ]
    subprocess.run(
        [str(PYTHON), "-u", "run_vmod_path_selection.py", *common],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    subprocess.run(
        [str(PYTHON), "-u", "run_vmod_path_confirmation.py", *common],
        cwd=ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
