"""Merge VMOD imitation datasets while preserving source provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ARRAY_KEYS = (
    "observations",
    "thetas",
    "target_actions",
    "day_indices",
    "candidate_ids",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_datasets(base_path: Path, augmentation_path: Path) -> dict[str, np.ndarray]:
    base = np.load(base_path)
    augmentation = np.load(augmentation_path)
    if set(base.files) != set(ARRAY_KEYS) or set(augmentation.files) != set(ARRAY_KEYS):
        raise ValueError("Both datasets must contain the canonical VMOD arrays")
    for key in ARRAY_KEYS:
        if base[key].ndim != augmentation[key].ndim:
            raise ValueError(f"Array rank mismatch for {key}")
        if base[key].shape[1:] != augmentation[key].shape[1:]:
            raise ValueError(f"Array shape mismatch for {key}")
    overlap = set(map(int, base["day_indices"])) & set(
        map(int, augmentation["day_indices"])
    )
    if overlap:
        raise ValueError(f"Base and augmentation day identifiers overlap: {sorted(overlap)}")
    return {
        key: np.concatenate((base[key], augmentation[key]), axis=0)
        for key in ARRAY_KEYS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--augmentation", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    arrays = merge_datasets(args.base, args.augmentation)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "opf_imitation_dataset.npz"
    np.savez_compressed(output, **arrays)
    manifest = {
        "role": "counterexample-guided feeder adaptation",
        "examples": int(len(arrays["observations"])),
        "observation_dim": int(arrays["observations"].shape[1]),
        "action_dim": int(arrays["target_actions"].shape[1]),
        "unique_days": int(np.unique(arrays["day_indices"]).size),
        "unique_candidates": int(np.unique(arrays["candidate_ids"]).size),
        "base": {
            "path": str(args.base),
            "sha256": file_sha256(args.base),
        },
        "augmentation": {
            "path": str(args.augmentation),
            "sha256": file_sha256(args.augmentation),
            "examples": int(len(np.load(args.augmentation)["observations"])),
        },
        "dataset_sha256": file_sha256(output),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
