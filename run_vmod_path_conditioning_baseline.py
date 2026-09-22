"""Extend the fixed-capacity VMOD ablation across boundary-adjacent points."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from run_vmod_conditioning_ablation import (
    contrast_summary,
    matched_fixed_dataset_indices,
    paired_seed_contrasts,
)
from vmod_current_study import (
    DEVELOPMENT_PROTOCOL_33,
    EXTERNAL_PROFILES_33,
    EXTERNAL_PROTOCOL_33,
    main_run_dir,
    read_selected_margin,
)


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
SEEDS = (42, 43, 44, 45, 46)


def parse_candidate_ids(value: str) -> tuple[int, ...]:
    """Parse a unique, ordered candidate list."""
    result: list[int] = []
    for item in value.split(","):
        if not item.strip():
            continue
        candidate_id = int(item)
        if candidate_id not in result:
            result.append(candidate_id)
    if not result:
        raise ValueError("At least one candidate id is required")
    return tuple(result)


def select_capacity_rows(
    capacity_path: pd.DataFrame, candidate_ids: tuple[int, ...]
) -> list[dict]:
    """Return requested path rows in caller order after schema validation."""
    required = {
        "path_index",
        "capacity_label",
        "pv_s_scale",
        "svc_q_scale",
        "cap_total_mvar",
    }
    missing = required - set(capacity_path.columns)
    if missing:
        raise ValueError(f"Capacity path is missing columns: {sorted(missing)}")
    indexed = capacity_path.set_index("path_index", drop=False)
    unknown = [candidate for candidate in candidate_ids if candidate not in indexed.index]
    if unknown:
        raise ValueError(f"Unknown candidate ids: {unknown}")
    return [indexed.loc[candidate].to_dict() for candidate in candidate_ids]


def process_environment(gpu: int | None = None) -> dict[str, str]:
    values = os.environ.copy()
    if gpu is not None:
        values["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        values[name] = "1"
    return values


def run(arguments: list[str], gpu: int | None = None) -> None:
    subprocess.run(
        [str(PYTHON), "-u", *arguments],
        cwd=ROOT,
        env=process_environment(gpu),
        check=True,
    )


def train_seed(seed: int, gpu: int, dataset: Path, policies: Path) -> None:
    checkpoint = policies / f"policy_init_seed{seed}.pth"
    if checkpoint.exists():
        return
    run(
        [
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
            str(DEVELOPMENT_PROTOCOL_33 / "student_validation_days.csv"),
            "--allow_missing_validation_days",
        ],
        gpu,
    )


def train_queue(
    gpu: int, seeds: tuple[int, ...], dataset: Path, policies: Path
) -> None:
    for seed in seeds:
        train_seed(seed, gpu, dataset, policies)


def build_fixed_dataset(
    source_path: Path,
    destination: Path,
    candidate_id: int,
    target_training_examples: int,
    target_validation_examples: int,
) -> dict:
    """Create an exactly budget-matched fixed-capacity imitation dataset."""
    source = np.load(source_path)
    mask = source["candidate_ids"] == int(candidate_id)
    if not np.any(mask):
        raise RuntimeError(f"No teacher examples for candidate {candidate_id}")
    validation_days = pd.read_csv(
        DEVELOPMENT_PROTOCOL_33 / "student_validation_days.csv"
    ).day_index.to_numpy(dtype=int)
    selected = matched_fixed_dataset_indices(
        source["day_indices"],
        mask,
        validation_days,
        target_training_examples,
        target_validation_examples,
        seed=20260921 + int(candidate_id),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        observations=source["observations"][selected],
        thetas=source["thetas"][selected],
        target_actions=source["target_actions"][selected],
        day_indices=source["day_indices"][selected],
        candidate_ids=source["candidate_ids"][selected],
    )
    available_days = sorted(set(map(int, source["day_indices"][mask])))
    available_validation = sorted(set(available_days) & set(map(int, validation_days)))
    manifest = {
        "candidate_id": int(candidate_id),
        "original_candidate_examples": int(mask.sum()),
        "original_candidate_days": len(available_days),
        "available_frozen_validation_days": available_validation,
        "missing_frozen_validation_days": sorted(
            set(map(int, validation_days)) - set(available_validation)
        ),
        "target_training_examples": int(target_training_examples),
        "target_validation_examples": int(target_validation_examples),
        "released_dataset_examples": int(len(selected)),
        "resampling_seed": 20260921 + int(candidate_id),
        "resampling": (
            "deterministic within-split oversampling to match the shared "
            "actor example and optimiser-step budget"
        ),
    }
    destination.with_name("dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def evaluate_policy(
    theta: list[float], policies: Path, split: str, destination: Path
) -> None:
    if (destination / "summary.json").exists():
        return
    run(
        [
            "evaluate_opf_initialisation_baseline.py",
            "--theta",
            ",".join(map(str, theta)),
            "--days_metadata",
            str(EXTERNAL_PROTOCOL_33 / f"{split}_days.csv"),
            "--profile_dir",
            str(EXTERNAL_PROFILES_33),
            "--initialisation_dir",
            str(policies),
            "--seeds",
            ",".join(map(str, SEEDS)),
            "--workers",
            "5",
            "--svc_absorption_ratio",
            "1.0",
            "--out_dir",
            str(destination),
        ]
    )


def train_budget(path_run: Path) -> tuple[int, int, list[dict]]:
    summaries = [
        json.loads(
            (path_run / "policies" / f"summary_seed{seed}.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in SEEDS
    ]
    training = {int(item["training_examples"]) for item in summaries}
    validation = {int(item["validation_examples"]) for item in summaries}
    if len(training) != 1 or len(validation) != 1:
        raise RuntimeError("Shared actors do not have a common training budget")
    return training.pop(), validation.pop(), summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate_ids",
        default="2,3,4",
        help="Comma-separated path indices; default covers rejected/selected/upper points.",
    )
    args = parser.parse_args()

    candidate_ids = parse_candidate_ids(args.candidate_ids)
    margin = read_selected_margin()
    path_run = main_run_dir(margin)
    capacity_path = pd.read_csv(EXTERNAL_PROTOCOL_33 / "capacity_path.csv")
    rows = select_capacity_rows(capacity_path, candidate_ids)
    target_training, target_validation, shared_summaries = train_budget(path_run)
    root = path_run / "path_conditioning_baseline"
    root.mkdir(parents=True, exist_ok=True)
    source_dataset = path_run / "dataset" / "opf_imitation_dataset.npz"

    point_summaries = []
    all_contrasts = []
    for row in rows:
        candidate_id = int(row["path_index"])
        label = str(row["capacity_label"])
        theta = [
            float(row["pv_s_scale"]),
            float(row["svc_q_scale"]),
            float(row["cap_total_mvar"]),
        ]
        point_root = root / label

        # The selected-point experiment is already complete and remains the
        # authoritative source for P03. Other points use identical settings.
        if candidate_id == 3 and (
            path_run / "conditioning_ablation" / "summary.json"
        ).exists():
            fixed_root = path_run / "conditioning_ablation"
            policies = fixed_root / "policies"
            dataset_manifest = json.loads(
                (fixed_root / "fixed_capacity_dataset_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
        else:
            fixed_root = point_root / "fixed"
            dataset = fixed_root / "fixed_capacity_dataset.npz"
            if not dataset.exists():
                dataset_manifest = build_fixed_dataset(
                    source_dataset,
                    dataset,
                    candidate_id,
                    target_training,
                    target_validation,
                )
            else:
                dataset_manifest = json.loads(
                    (fixed_root / "dataset_manifest.json").read_text(encoding="utf-8")
                )
            policies = fixed_root / "policies"
            policies.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(train_queue, 0, (42, 44, 46), dataset, policies),
                    executor.submit(train_queue, 1, (43, 45), dataset, policies),
                ]
                for future in futures:
                    future.result()

        fixed_summaries = [
            json.loads((policies / f"summary_seed{seed}.json").read_text(encoding="utf-8"))
            for seed in SEEDS
        ]
        for shared_summary, fixed_summary in zip(shared_summaries, fixed_summaries):
            if int(shared_summary["training_examples"]) != int(
                fixed_summary["training_examples"]
            ) or int(shared_summary["training_example_updates"]) != int(
                fixed_summary["training_example_updates"]
            ):
                raise RuntimeError(f"Training budgets differ at {label}")

        split_summaries = {}
        for split in ("selection", "confirmation"):
            shared_out = point_root / "shared" / split
            fixed_out = point_root / "fixed_evaluation" / split
            frozen_shared_daily = path_run / split / label / "daily.csv"
            if frozen_shared_daily.exists():
                shared_daily = frozen_shared_daily
            else:
                evaluate_policy(theta, path_run / "policies", split, shared_out)
                shared_daily = shared_out / "daily.csv"
            if candidate_id == 3 and (fixed_root / split / "daily.csv").exists():
                fixed_daily = fixed_root / split / "daily.csv"
            else:
                evaluate_policy(theta, policies, split, fixed_out)
                fixed_daily = fixed_out / "daily.csv"
            shared = pd.read_csv(shared_daily)
            fixed = pd.read_csv(fixed_daily)
            contrasts = paired_seed_contrasts(shared, fixed, split)
            split_summaries[split] = contrast_summary(contrasts)
            split_summaries[split].update(
                {
                    "shared_total_event_days": int(shared.event.sum()),
                    "fixed_total_event_days": int(fixed.event.sum()),
                    "shared_all_seeds_zero_events": bool(
                        (shared.groupby("seed").event.sum() == 0).all()
                    ),
                    "fixed_all_seeds_zero_events": bool(
                        (fixed.groupby("seed").event.sum() == 0).all()
                    ),
                }
            )
            all_contrasts.extend(
                {"candidate_id": candidate_id, "capacity_label": label, **item}
                for item in contrasts
            )

        point_summaries.append(
            {
                "candidate_id": candidate_id,
                "capacity_label": label,
                "theta": theta,
                "dataset_manifest": dataset_manifest,
                "training_budget_matched": True,
                "split_summaries": split_summaries,
            }
        )

    summary = {
        "candidate_ids": list(candidate_ids),
        "comparison": "shared capacity-conditioned actor versus fixed-capacity actors",
        "training_examples_per_seed": target_training,
        "validation_examples_per_seed": target_validation,
        "training_example_updates_per_seed": int(
            shared_summaries[0]["training_example_updates"]
        ),
        "seeds": list(SEEDS),
        "points": point_summaries,
    }
    pd.DataFrame(all_contrasts).to_csv(root / "seed_level_contrasts.csv", index=False)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
