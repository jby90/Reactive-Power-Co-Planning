"""Generate, distil, and calibrate one VMOD voltage-margin candidate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
PROFILES = ROOT / "data" / "vmod" / "profiles33"
JOBS = ROOT / "runs" / "CSTC_OPF_GATE_20260920" / "training" / "jobs.csv"
CALIBRATION_DAYS = ROOT / "runs" / "VMOD_PROTOCOL_20260920" / "calibration_days.csv"
SEEDS = (42, 43, 44, 45, 46)


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


def pretrain(
    seed: int,
    gpu: int,
    dataset: Path,
    policies: Path,
    validation_days: Path,
) -> None:
    checkpoint = policies / f"policy_init_seed{seed}.pth"
    if checkpoint.exists():
        return
    run(
        [
            str(PYTHON),
            "-u",
            "pretrain_policy_from_opf.py",
            "--dataset",
            str(dataset),
            "--out_dir",
            str(policies),
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
            str(validation_days),
        ],
        gpu,
    )


def train_queue(
    gpu: int,
    seeds: tuple[int, ...],
    dataset: Path,
    policies: Path,
    validation_days: Path,
) -> None:
    for seed in seeds:
        pretrain(seed, gpu, dataset, policies, validation_days)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin", type=float, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--teacher_dir", type=Path)
    parser.add_argument("--reuse_dataset", type=Path)
    parser.add_argument("--teacher_workers", type=int, default=2)
    parser.add_argument("--dataset_workers", type=int, default=4)
    parser.add_argument("--svc_absorption_ratio", type=float, default=0.0)
    parser.add_argument("--jobs_csv", type=Path, default=JOBS)
    parser.add_argument("--calibration_days", type=Path, default=CALIBRATION_DAYS)
    parser.add_argument(
        "--student_validation_days",
        type=Path,
        default=CALIBRATION_DAYS,
        help="Internal early-stopping days that must be present in the fit dataset.",
    )
    parser.add_argument("--no_direct_fallback", action="store_true")
    parser.add_argument("--calibration_theta", default="0.45,0.375,0")
    parser.add_argument(
        "--calibration_label",
        default="P04_pv045_svc0375",
        help="Frozen feasible-reference point used only to select the voltage margin.",
    )
    args = parser.parse_args()

    out = args.out_dir.resolve()
    teacher = (args.teacher_dir or (out / "teacher")).resolve()
    dataset_dir = out / "dataset"
    dataset = dataset_dir / "opf_imitation_dataset.npz"
    policies = out / "policies"
    calibration = out / f"calibration_{args.calibration_label}"
    out.mkdir(parents=True, exist_ok=True)
    jobs_csv = args.jobs_csv.resolve()
    calibration_days = args.calibration_days.resolve()
    student_validation_days = args.student_validation_days.resolve()

    if not (teacher / "summary.json").exists():
        teacher_command = [
            str(PYTHON),
            "-u",
            "ac_opf_nested_envelope.py",
            "--jobs_csv",
            str(jobs_csv),
            "--out_dir",
            str(teacher),
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
            str(args.svc_absorption_ratio),
        ]
        if args.no_direct_fallback:
            teacher_command.append("--no_direct_fallback")
        run(teacher_command)

    if not dataset.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)
        if args.reuse_dataset is not None:
            source = args.reuse_dataset.resolve()
            shutil.copy2(source, dataset)
            source_manifest = source.parent / "manifest.json"
            if source_manifest.exists():
                shutil.copy2(source_manifest, dataset_dir / "manifest.json")
        else:
            run(
                [
                    str(PYTHON),
                    "-u",
                    "build_opf_imitation_dataset.py",
                    "--jobs_csv",
                    str(jobs_csv),
                    "--opf_steps_csv",
                    str(teacher / "steps.csv"),
                    "--load_profile",
                    str(PROFILES / "load_15min_366d.npy"),
                    "--generation_profile",
                    str(PROFILES / "pv_15min_366d.npy"),
                    "--out_dir",
                    str(dataset_dir),
                    "--workers",
                    str(args.dataset_workers),
                    "--svc_absorption_ratio",
                    str(args.svc_absorption_ratio),
                ]
            )

    policies.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                train_queue,
                0,
                (42, 44, 46),
                dataset,
                policies,
                student_validation_days,
            ),
            executor.submit(
                train_queue,
                1,
                (43, 45),
                dataset,
                policies,
                student_validation_days,
            ),
        ]
        for future in futures:
            future.result()

    if not (calibration / "summary.json").exists():
        run(
            [
                str(PYTHON),
                "-u",
                "evaluate_opf_initialisation_baseline.py",
                "--theta",
                args.calibration_theta,
                "--days_metadata",
                str(calibration_days),
                "--profile_dir",
                str(PROFILES),
                "--initialisation_dir",
                str(policies),
                "--seeds",
                ",".join(map(str, SEEDS)),
                "--workers",
                "5",
                "--svc_absorption_ratio",
                str(args.svc_absorption_ratio),
                "--out_dir",
                str(calibration),
            ]
        )

    evaluation = json.loads((calibration / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    calibration_support = next(
        row
        for row in manifest["candidate_coverage"]
        if row["capacity_label"] == args.calibration_label
    )
    coverage_fraction = (
        int(calibration_support["converged_days"])
        / int(calibration_support["attempted_days"])
    )
    result = {
        "margin_pu": args.margin,
        "teacher_voltage_band": [0.95 + args.margin, 1.05 - args.margin],
        "svc_absorption_ratio": args.svc_absorption_ratio,
        "fit_days": 40,
        "calibration_days": 10,
        "calibration_reference_label": args.calibration_label,
        "calibration_reference_theta": args.calibration_theta,
        "all_seeds_zero_events": evaluation["all_seeds_zero_events"],
        "all_seeds_zero_pf_failures": evaluation.get(
            "all_seeds_zero_pf_failures",
            all(int(row["pf_fail_days"]) == 0 for row in evaluation["seed_summaries"]),
        ),
        "total_event_days": evaluation["total_event_days"],
        "total_pf_fail_days": evaluation.get(
            "total_pf_fail_days",
            sum(int(row["pf_fail_days"]) for row in evaluation["seed_summaries"]),
        ),
        "maximum_seed_event_days": evaluation["maximum_seed_event_days"],
        "mean_seed_loss_mwh": evaluation["mean_seed_loss_mwh"],
        "minimum_seed_voltage_pu": min(
            row["worst_minimum_voltage_pu"] for row in evaluation["seed_summaries"]
        ),
        "maximum_seed_voltage_pu": max(
            row["worst_maximum_voltage_pu"] for row in evaluation["seed_summaries"]
        ),
        "dataset_sha256": manifest["dataset_sha256"],
        "jobs_csv": str(jobs_csv),
        "calibration_days_csv": str(calibration_days),
        "student_validation_days_csv": str(student_validation_days),
        "calibration_reference_teacher_attempted_days": int(
            calibration_support["attempted_days"]
        ),
        "calibration_reference_teacher_converged_days": int(
            calibration_support["converged_days"]
        ),
        "calibration_reference_teacher_coverage_fraction": coverage_fraction,
        "adequate_calibration_teacher_support": coverage_fraction >= 0.8,
        "direct_fallback_allowed": not args.no_direct_fallback,
        "teacher_dir": str(teacher),
    }
    (out / "calibration_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
