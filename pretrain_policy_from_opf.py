"""Pretrain the shared capacity-conditioned actor on AC-OPF demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from vmod_actor import CapacityConditionedActor


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_frozen_validation_days(
    requested_days: np.ndarray,
    dataset_days: np.ndarray,
    *,
    allow_missing: bool,
) -> tuple[np.ndarray, list[int]]:
    """Keep the frozen split while reporting unavailable teacher trajectories.

    A missing validation day is never reassigned to training.  The portability
    study may continue with the available subset only when this behaviour is
    explicitly requested; the principal study remains strict by default.
    """
    requested = np.asarray(sorted(set(map(int, requested_days))), dtype=int)
    available = set(map(int, dataset_days))
    missing = sorted(set(map(int, requested)) - available)
    if missing and not allow_missing:
        raise ValueError(f"Validation days are absent from the dataset: {missing}")
    realised = np.asarray(
        [day for day in requested if int(day) in available], dtype=int
    )
    if len(realised) == 0 or len(realised) >= len(available):
        raise ValueError("Validation split must leave available validation and training days")
    return realised, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--validation_days", type=int, default=10)
    parser.add_argument(
        "--validation_days_csv",
        type=Path,
        help="Optional frozen day_index list shared by every training seed.",
    )
    parser.add_argument(
        "--allow_missing_validation_days",
        action="store_true",
        help=(
            "Continue with the available subset of a frozen validation list when "
            "teacher trajectories are absent. Missing days remain excluded and are "
            "reported; they are never reassigned to training."
        ),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    data = np.load(args.dataset)
    observations = data["observations"]
    thetas = data["thetas"]
    targets = data["target_actions"]
    day_indices = data["day_indices"]
    unique_days = np.unique(day_indices)
    rng = np.random.default_rng(args.seed + 1909)
    requested_validation_days: np.ndarray
    missing_validation_days: list[int]
    if args.validation_days_csv is not None:
        validation_frame = pd.read_csv(args.validation_days_csv)
        if "day_index" not in validation_frame:
            raise ValueError("--validation_days_csv must contain day_index")
        requested_validation_days = np.asarray(
            sorted(set(validation_frame.day_index.astype(int))), dtype=int
        )
        validation_days, missing_validation_days = resolve_frozen_validation_days(
            requested_validation_days,
            unique_days,
            allow_missing=args.allow_missing_validation_days,
        )
    else:
        validation_days = rng.choice(
            unique_days,
            size=min(args.validation_days, len(unique_days) - 1),
            replace=False,
        )
        requested_validation_days = np.asarray(validation_days, dtype=int)
        missing_validation_days = []
    validation_mask = np.isin(day_indices, validation_days)
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CapacityConditionedActor(
        observations.shape[1],
        targets.shape[1],
        [0.225, 0.0, 0.0],
        [1.5, 1.5, 1.0],
        log_std_init=-2.0,
    ).to(device)
    optimiser = torch.optim.AdamW(
        model.policy_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
    theta_tensor = torch.as_tensor(thetas, dtype=torch.float32)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32)
    rows = []
    training_start = perf_counter()
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        permutation = rng.permutation(train_indices)
        model.train()
        train_losses = []
        for start in range(0, len(permutation), args.batch_size):
            batch = permutation[start : start + args.batch_size]
            obs = obs_tensor[batch].to(device)
            theta = theta_tensor[batch].to(device)
            target = target_tensor[batch].to(device)
            prediction, _ = model._pi(obs, theta)
            loss = torch.square(prediction - target).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            prediction, _ = model._pi(
                obs_tensor[validation_indices].to(device),
                theta_tensor[validation_indices].to(device),
            )
            validation_loss = float(
                torch.square(
                    prediction - target_tensor[validation_indices].to(device)
                ).mean().cpu()
            )
        rows.append(
            {
                "epoch": epoch,
                "training_mse": float(np.mean(train_losses)),
                "validation_mse": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if key.startswith(("pi_fc1.", "pi_fc2.", "pi_mu.", "pi_log_std"))
            }
    assert best_state is not None
    training_wall_seconds = perf_counter() - training_start
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / f"policy_init_seed{args.seed}.pth"
    torch.save(
        {
            "policy_state_dict": best_state,
            "seed": args.seed,
            "dataset_sha256": file_hash(args.dataset),
            "best_validation_mse": best_loss,
            "validation_days": sorted(int(day) for day in validation_days),
            "requested_validation_days": sorted(
                int(day) for day in requested_validation_days
            ),
            "missing_validation_days": missing_validation_days,
            "config": vars(args),
        },
        checkpoint,
    )
    pd.DataFrame(rows).to_csv(
        args.out_dir / f"training_seed{args.seed}.csv", index=False
    )
    summary = {
        "seed": args.seed,
        "training_examples": int(len(train_indices)),
        "validation_examples": int(len(validation_indices)),
        "dataset_unique_days_with_examples": int(len(unique_days)),
        "training_unique_days_with_examples": int(
            len(np.unique(day_indices[train_indices]))
        ),
        "validation_unique_days_with_examples": int(
            len(np.unique(day_indices[validation_indices]))
        ),
        "validation_days": sorted(int(day) for day in validation_days),
        "requested_validation_days": sorted(
            int(day) for day in requested_validation_days
        ),
        "missing_validation_days": missing_validation_days,
        "validation_day_policy": (
            "frozen_available_subset_without_reassignment"
            if missing_validation_days
            else "frozen_complete"
        ),
        "best_validation_mse": best_loss,
        "best_epoch": int(pd.DataFrame(rows).validation_mse.idxmin() + 1),
        "dataset_sha256": file_hash(args.dataset),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint),
        "device": str(device),
        "training_wall_seconds": training_wall_seconds,
        "training_example_updates": int(len(train_indices) * args.epochs),
        "training_example_updates_per_second": float(
            len(train_indices) * args.epochs / training_wall_seconds
        ),
        "trainable_policy_parameters": int(
            sum(parameter.numel() for parameter in model.policy_parameters())
        ),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }
    (args.out_dir / f"summary_seed{args.seed}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
