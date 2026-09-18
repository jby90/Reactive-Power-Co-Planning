"""Verify release hashes, profile splits, and principal checkpoint coverage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest() -> int:
    manifest_path = ROOT / "data" / "release_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = set()
    for row in payload["files"]:
        relative = Path(row["path"])
        listed.add(relative.as_posix())
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        if path.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"Size mismatch: {relative}")
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {relative}")
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path != manifest_path
        and "__pycache__" not in path.parts
        and "reproduced" not in path.parts
    }
    missing = sorted(actual - listed)
    stale = sorted(listed - actual)
    if missing or stale:
        raise RuntimeError(f"Manifest coverage mismatch; unlisted={missing}, stale={stale}")
    return len(listed)


def verify_inputs() -> tuple[int, int, int]:
    input_dir = ROOT / "data" / "inputs"
    splits = {
        name: set(pd.read_csv(input_dir / f"{name}_days.csv")["day_index"].astype(int))
        for name in ("training", "selection", "confirmation")
    }
    if any(splits[a] & splits[b] for a, b in (("training", "selection"), ("training", "confirmation"), ("selection", "confirmation"))):
        raise RuntimeError("Frozen day splits overlap")
    if set().union(*splits.values()) != set(range(366)):
        raise RuntimeError("Frozen day splits do not cover all 366 days")
    load = np.load(input_dir / "load_15min_366d.npy", mmap_mode="r")
    pv = np.load(input_dir / "pv_15min_366d.npy", mmap_mode="r")
    if load.shape != (366 * 96, 32) or pv.shape != (366 * 96, 3):
        raise RuntimeError(f"Unexpected profile shapes: load={load.shape}, pv={pv.shape}")
    return tuple(len(splits[name]) for name in ("training", "selection", "confirmation"))


def verify_checkpoints() -> int:
    principal = []
    for method in ("WG_CVAR_PPO", "CONCAT_SCALAR_CORRECTED"):
        for seed in range(42, 47):
            matches = list((ROOT / "models" / method).glob(f"env33_seed{seed}_gamma0.9/*confirm-opsd*seed{seed}/models/checkpoint.pth"))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one principal checkpoint for {method}, seed {seed}; found {len(matches)}")
            principal.extend(matches)
    return len(principal)


def main() -> None:
    files = verify_manifest()
    split_counts = verify_inputs()
    checkpoints = verify_checkpoints()
    print(
        f"Release verified: {files} files, split={split_counts}, "
        f"principal checkpoints={checkpoints}."
    )


if __name__ == "__main__":
    main()
