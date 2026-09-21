"""Distil and calibrate the frozen nine-level VMOD capacity path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from vmod_evaluation import action_reference_scales


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
OUT = ROOT / "runs" / "VMOD_PATH_MARGIN005_20260920"
TEACHER = OUT / "teacher"
DATASET_DIR = OUT / "dataset"
DATASET = DATASET_DIR / "opf_imitation_dataset.npz"
POLICIES = OUT / "policies"
PROTOCOL = ROOT / "runs" / "VMOD_PROTOCOL_20260920"
PROFILES = ROOT / "data" / "vmod" / "profiles33"
SEEDS = (42, 43, 44, 45, 46)
SVC_ABSORPTION_RATIO = 0.0
ALLOW_MISSING_VALIDATION_DAYS = False


def environment(gpu: int) -> dict[str, str]:
    values = os.environ.copy()
    values.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return values


def run(command: list[str], gpu: int = 0) -> None:
    subprocess.run(command, cwd=ROOT, env=environment(gpu), check=True)


def pretrain(seed: int, gpu: int) -> None:
    checkpoint = POLICIES / f"policy_init_seed{seed}.pth"
    if checkpoint.exists():
        return
    command = [
            str(PYTHON),
            "-u",
            "pretrain_policy_from_opf.py",
            "--dataset",
            str(DATASET),
            "--out_dir",
            str(POLICIES),
            "--seed",
            str(seed),
            "--epochs",
            "300",
            "--batch_size",
            "512",
            "--learning_rate",
            "0.001",
            "--weight_decay",
            "0.00001",
            "--validation_days_csv",
            str(PROTOCOL / "student_validation_days.csv"),
        ]
    if ALLOW_MISSING_VALIDATION_DAYS:
        command.append("--allow_missing_validation_days")
    run(command, gpu)


def train_queue(gpu: int, seeds: tuple[int, ...]) -> None:
    for seed in seeds:
        pretrain(seed, gpu)


def main() -> None:
    global OUT, TEACHER, DATASET_DIR, DATASET, POLICIES, PROTOCOL, PROFILES
    global SVC_ABSORPTION_RATIO, ALLOW_MISSING_VALIDATION_DAYS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=OUT)
    parser.add_argument("--teacher_dir", type=Path)
    parser.add_argument("--protocol_dir", type=Path, default=PROTOCOL)
    parser.add_argument("--profile_dir", type=Path, default=PROFILES)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--svc_absorption_ratio", type=float, default=0.0)
    parser.add_argument("--env", type=int, default=33, choices=[33, 69, 118])
    parser.add_argument("--cap_buses", default="20,8")
    parser.add_argument("--build_teacher", action="store_true")
    parser.add_argument("--teacher_workers", type=int, default=4)
    parser.add_argument("--dataset_workers", type=int, default=6)
    parser.add_argument(
        "--allow_missing_validation_days",
        action="store_true",
        help=(
            "Use only available days from the frozen validation list when teacher "
            "trajectories are missing; no day is reassigned."
        ),
    )
    args = parser.parse_args()

    OUT = args.out_dir.resolve()
    TEACHER = (args.teacher_dir or (OUT / "teacher")).resolve()
    DATASET_DIR = OUT / "dataset"
    DATASET = DATASET_DIR / "opf_imitation_dataset.npz"
    POLICIES = OUT / "policies"
    PROTOCOL = args.protocol_dir.resolve()
    PROFILES = args.profile_dir.resolve()
    SVC_ABSORPTION_RATIO = float(args.svc_absorption_ratio)
    ALLOW_MISSING_VALIDATION_DAYS = bool(args.allow_missing_validation_days)

    if args.build_teacher and not (TEACHER / "summary.json").exists():
        run(
            [
                str(PYTHON),
                "-u",
                "ac_opf_nested_envelope.py",
                "--jobs_csv",
                str(PROTOCOL / "training_path_jobs.csv"),
                "--out_dir",
                str(TEACHER),
                "--workers",
                str(args.teacher_workers),
                "--lower_voltage",
                str(0.95 + args.margin),
                "--upper_voltage",
                str(1.05 - args.margin),
                "--load_profile",
                str(PROFILES / "load_15min_366d.npy"),
                "--generation_profile",
                str(PROFILES / "pv_15min_366d.npy"),
                "--svc_absorption_ratio",
                str(SVC_ABSORPTION_RATIO),
                "--env",
                str(args.env),
                "--cap_buses",
                args.cap_buses,
                "--no_direct_fallback",
            ]
        )
    while not (TEACHER / "summary.json").exists():
        time.sleep(30)
    if not DATASET.exists():
        run(
            [
                str(PYTHON),
                "-u",
                "build_opf_imitation_dataset.py",
                "--jobs_csv",
                str(PROTOCOL / "training_path_jobs.csv"),
                "--opf_steps_csv",
                str(TEACHER / "steps.csv"),
                "--load_profile",
                str(PROFILES / "load_15min_366d.npy"),
                "--generation_profile",
                str(PROFILES / "pv_15min_366d.npy"),
                "--out_dir",
                str(DATASET_DIR),
                "--workers",
                str(args.dataset_workers),
                "--svc_absorption_ratio",
                str(SVC_ABSORPTION_RATIO),
                "--env",
                str(args.env),
            ]
        )

    POLICIES.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(train_queue, 0, (42, 44, 46)),
            executor.submit(train_queue, 1, (43, 45)),
        ]
        for future in futures:
            future.result()

    path = pd.read_csv(PROTOCOL / "capacity_path.csv").sort_values("path_index")
    rows = []
    for point in path.itertuples(index=False):
        out_dir = OUT / "calibration" / point.capacity_label
        if not (out_dir / "summary.json").exists():
            run(
                [
                    str(PYTHON),
                    "-u",
                    "evaluate_opf_initialisation_baseline.py",
                    "--theta",
                    f"{point.pv_s_scale},{point.svc_q_scale},{point.cap_total_mvar}",
                    "--days_metadata",
                    str(PROTOCOL / "calibration_days.csv"),
                    "--profile_dir",
                    str(PROFILES),
                    "--initialisation_dir",
                    str(POLICIES),
                    "--seeds",
                    ",".join(map(str, SEEDS)),
                    "--workers",
                    "5",
                    "--svc_absorption_ratio",
                    str(SVC_ABSORPTION_RATIO),
                    "--env",
                    str(args.env),
                    "--cap_buses",
                    args.cap_buses,
                    "--out_dir",
                    str(out_dir),
                ]
            )
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        rows.append(
            {
                **point._asdict(),
                "evaluated_seed_days": 5 * summary["evaluated_days_per_seed"],
                "all_seeds_zero_events": summary["all_seeds_zero_events"],
                "total_event_days": summary["total_event_days"],
                "maximum_seed_event_days": summary["maximum_seed_event_days"],
                "mean_seed_loss_mwh": summary["mean_seed_loss_mwh"],
                "worst_vmin": min(
                    item["worst_minimum_voltage_pu"]
                    for item in summary["seed_summaries"]
                ),
                "worst_vmax": max(
                    item["worst_maximum_voltage_pu"]
                    for item in summary["seed_summaries"]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "calibration_path_summary.csv", index=False)
    metadata = {
        "margin_pu": args.margin,
        "svc_absorption_ratio": SVC_ABSORPTION_RATIO,
        "environment": args.env,
        "reference_pv_s_scale": action_reference_scales(args.env)[0],
        "reference_svc_q_scale": action_reference_scales(args.env)[1],
        "capacitor_buses": args.cap_buses,
        "fit_days": 40,
        "calibration_days": 10,
        "capacity_points": len(frame),
    }
    (OUT / "path_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
