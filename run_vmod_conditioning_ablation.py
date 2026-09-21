"""Compare shared capacity conditioning with fixed-capacity VMOD students."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from vmod_statistics import mean_t_interval
from vmod_current_study import (
    DEVELOPMENT_PROTOCOL_33,
    EXTERNAL_PROFILES_33,
    EXTERNAL_PROTOCOL_33,
    main_run_dir,
    read_selected_margin,
)


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
PROTOCOL = EXTERNAL_PROTOCOL_33
PROFILES = EXTERNAL_PROFILES_33
SEEDS = (42, 43, 44, 45, 46)


def resample_to_count(
    indices: np.ndarray, target_count: int, rng: np.random.Generator
) -> np.ndarray:
    """Deterministically oversample an index pool to an exact budget."""
    pool = np.asarray(indices, dtype=int)
    if pool.size == 0:
        raise ValueError("Cannot resample an empty example pool")
    if target_count <= 0:
        raise ValueError("Target example count must be positive")
    repetitions, remainder = divmod(int(target_count), int(pool.size))
    parts = [np.tile(pool, repetitions)] if repetitions else []
    if remainder:
        parts.append(rng.choice(pool, size=remainder, replace=False))
    result = np.concatenate(parts)
    rng.shuffle(result)
    return result


def matched_fixed_dataset_indices(
    day_indices: np.ndarray,
    candidate_mask: np.ndarray,
    validation_days: np.ndarray,
    target_training_examples: int,
    target_validation_examples: int,
    seed: int = 20260921,
) -> np.ndarray:
    """Match shared-model train/validation budgets without adding information."""
    all_days = np.asarray(day_indices, dtype=int)
    candidate_indices = np.flatnonzero(np.asarray(candidate_mask, dtype=bool))
    candidate_days = all_days[candidate_indices]
    validation_mask = np.isin(candidate_days, np.asarray(validation_days, dtype=int))
    training_pool = candidate_indices[~validation_mask]
    validation_pool = candidate_indices[validation_mask]
    rng = np.random.default_rng(seed)
    training = resample_to_count(training_pool, target_training_examples, rng)
    validation = resample_to_count(validation_pool, target_validation_examples, rng)
    return np.concatenate((training, validation))


def environment(gpu: int) -> dict[str, str]:
    values = os.environ.copy()
    values["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        values[name] = "1"
    return values


def run(command: list[str], gpu: int = 0) -> None:
    subprocess.run(command, cwd=ROOT, env=environment(gpu), check=True)


def train(seed: int, gpu: int, dataset: Path, policies: Path) -> None:
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
            str(DEVELOPMENT_PROTOCOL_33 / "student_validation_days.csv"),
        ],
        gpu,
    )


def train_queue(gpu: int, seeds: tuple[int, ...], dataset: Path, policies: Path) -> None:
    for seed in seeds:
        train(seed, gpu, dataset, policies)


def paired_seed_contrasts(
    shared: pd.DataFrame, fixed: pd.DataFrame, split: str
) -> list[dict]:
    """Build seed-level contrasts on an exactly matched operating-day set."""
    contrasts = []
    for seed in SEEDS:
        shared_seed = shared.query("seed == @seed")
        fixed_seed = fixed.query("seed == @seed")
        paired = shared_seed.merge(
            fixed_seed,
            on="day",
            suffixes=("_shared", "_fixed"),
            validate="one_to_one",
        )
        if len(paired) != len(shared_seed) or len(paired) != len(fixed_seed):
            raise RuntimeError(
                f"Seed {seed} does not have an exactly matched {split} day set"
            )
        contrasts.append(
            {
                "split": split,
                "seed": seed,
                "paired_days": len(paired),
                "shared_event_days": int(paired.event_shared.sum()),
                "fixed_event_days": int(paired.event_fixed.sum()),
                "shared_minus_fixed_loss_mwh": float(
                    (paired.line_loss_mwh_shared - paired.line_loss_mwh_fixed).mean()
                ),
            }
        )
    return contrasts


def contrast_summary(contrasts: list[dict]) -> dict:
    frame = pd.DataFrame(contrasts)
    mean, low, high = mean_t_interval(
        frame.shared_minus_fixed_loss_mwh.to_numpy()
    )
    return {
        "shared_minus_fixed_loss_mean_mwh": mean,
        "seed_level_t_ci95_low_mwh": low,
        "seed_level_t_ci95_high_mwh": high,
        "seed_contrasts": contrasts,
    }


def main() -> None:
    margin = read_selected_margin()
    path_run = main_run_dir(margin)
    boundary_file = path_run / "confirmation_boundary_summary.json"
    while not boundary_file.exists():
        time.sleep(60)
    boundary = json.loads(boundary_file.read_text(encoding="utf-8"))
    if not boundary["boundary_confirmed"]:
        raise RuntimeError("The main capacity boundary did not pass confirmation")
    selected = next(
        point for point in boundary["points"] if point["role"] == "selected"
    )
    candidate_id = int(selected["path_index"])
    capacity_label = str(selected["capacity_label"])
    theta = [
        float(selected["pv_s_scale"]),
        float(selected["svc_q_scale"]),
        float(selected["cap_total_mvar"]),
    ]

    out = path_run / "conditioning_ablation"
    dataset = out / "fixed_capacity_dataset.npz"
    shared_summaries = [
        json.loads(
            (path_run / "policies" / f"summary_seed{seed}.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in SEEDS
    ]
    shared_training_counts = {
        int(row["training_examples"]) for row in shared_summaries
    }
    shared_validation_counts = {
        int(row["validation_examples"]) for row in shared_summaries
    }
    if len(shared_training_counts) != 1 or len(shared_validation_counts) != 1:
        raise RuntimeError("Shared actors do not have a common example budget")
    target_training_examples = shared_training_counts.pop()
    target_validation_examples = shared_validation_counts.pop()
    if not dataset.exists():
        source = np.load(path_run / "dataset" / "opf_imitation_dataset.npz")
        mask = source["candidate_ids"] == candidate_id
        if not np.any(mask):
            raise RuntimeError(f"No training examples for candidate {candidate_id}")
        validation_days = pd.read_csv(
            DEVELOPMENT_PROTOCOL_33 / "student_validation_days.csv"
        ).day_index.to_numpy(dtype=int)
        selected_indices = matched_fixed_dataset_indices(
            source["day_indices"],
            mask,
            validation_days,
            target_training_examples,
            target_validation_examples,
        )
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dataset,
            observations=source["observations"][selected_indices],
            thetas=source["thetas"][selected_indices],
            target_actions=source["target_actions"][selected_indices],
            day_indices=source["day_indices"][selected_indices],
            candidate_ids=source["candidate_ids"][selected_indices],
        )
        manifest = {
            "candidate_id": candidate_id,
            "original_candidate_examples": int(mask.sum()),
            "target_training_examples": target_training_examples,
            "target_validation_examples": target_validation_examples,
            "released_dataset_examples": int(len(selected_indices)),
            "resampling": (
                "deterministic within-split oversampling to match the shared "
                "actor's example and optimizer-step budget"
            ),
            "resampling_seed": 20260921,
        }
        (out / "fixed_capacity_dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    policies = out / "policies"
    policies.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(train_queue, 0, (42, 44, 46), dataset, policies),
            executor.submit(train_queue, 1, (43, 45), dataset, policies),
        ]
        for future in futures:
            future.result()

    evaluations = {}
    for split in ("selection", "confirmation"):
        evaluation = out / split
        evaluations[split] = evaluation
        if not (evaluation / "summary.json").exists():
            run(
                [
                    str(PYTHON),
                    "-u",
                    "evaluate_opf_initialisation_baseline.py",
                    "--theta",
                    ",".join(map(str, theta)),
                    "--days_metadata",
                    str(PROTOCOL / f"{split}_days.csv"),
                    "--profile_dir",
                    str(PROFILES),
                    "--initialisation_dir",
                    str(policies),
                    "--seeds",
                    ",".join(map(str, SEEDS)),
                    "--workers",
                    "5",
                    "--svc_absorption_ratio",
                    "1.0",
                    "--out_dir",
                    str(evaluation),
                ]
            )

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
            raise RuntimeError("Conditioning ablation training budgets are not matched")
    split_summaries = {}
    all_contrasts = []
    for split in ("selection", "confirmation"):
        shared = pd.read_csv(
            path_run / split / capacity_label / "daily.csv"
        )
        fixed = pd.read_csv(evaluations[split] / "daily.csv")
        contrasts = paired_seed_contrasts(shared, fixed, split)
        split_summaries[split] = contrast_summary(contrasts)
        all_contrasts.extend(contrasts)
    primary = split_summaries["confirmation"]
    summary = {
        "selected_path_index": candidate_id,
        "capacity_label": capacity_label,
        "theta": theta,
        "shared_training_capacity_points": 9,
        "fixed_training_capacity_points": 1,
        "shared_training_examples_per_seed": target_training_examples,
        "fixed_training_examples_per_seed": int(
            fixed_summaries[0]["training_examples"]
        ),
        "training_example_updates_per_seed": int(
            fixed_summaries[0]["training_example_updates"]
        ),
        "training_budget_matched": True,
        "n_training_seeds": len(SEEDS),
        "primary_inference_split": "confirmation",
        "primary_split_reason": (
            "The 349 confirmation days did not participate in capacity-point "
            "selection; the 17-day selection comparison is retained as a diagnostic."
        ),
        "shared_minus_fixed_loss_mean_mwh": primary[
            "shared_minus_fixed_loss_mean_mwh"
        ],
        "seed_level_t_ci95_low_mwh": primary["seed_level_t_ci95_low_mwh"],
        "seed_level_t_ci95_high_mwh": primary["seed_level_t_ci95_high_mwh"],
        "seed_contrasts": primary["seed_contrasts"],
        "split_summaries": split_summaries,
    }
    pd.DataFrame(all_contrasts).to_csv(
        out / "seed_level_contrasts.csv", index=False
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
