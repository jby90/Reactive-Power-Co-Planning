"""Generate immutable, disjoint aligned-day splits for final capacity planning."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "inputs"
OUT_DIR = ROOT / "reproduced" / "capacity_planning" / "locked_days"
SEED = 2026071701
SPLIT_SIZES = {"screen": 3, "refine": 30, "confirm": 200}
STEPS_PER_DAY = 96


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    load = np.load(DATA_DIR / "load96.npy")
    generation = np.load(DATA_DIR / "gen96.npy")
    total_days = min(len(load), len(generation)) // STEPS_PER_DAY
    safe_days = np.arange(max(1, total_days - 1), dtype=int)
    required = sum(SPLIT_SIZES.values())
    if required > len(safe_days):
        raise ValueError(f"Need {required} unique days but only {len(safe_days)} are safe")

    rng = np.random.default_rng(SEED)
    selected = rng.choice(safe_days, size=required, replace=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "steps_per_day": STEPS_PER_DAY,
        "total_profile_days": total_days,
        "safe_day_count": len(safe_days),
        "splits": {},
    }
    offset = 0
    all_days: list[int] = []
    for name, size in SPLIT_SIZES.items():
        days = np.sort(selected[offset : offset + size])
        offset += size
        all_days.extend(days.tolist())
        frame = pd.DataFrame(
            {
                "scenario_index": np.arange(size, dtype=int),
                "day_index": days,
                "t0": days * STEPS_PER_DAY,
            }
        )
        path = OUT_DIR / f"{name}_days.csv"
        frame.to_csv(path, index=False)
        metadata["splits"][name] = {
            "rows": size,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "day_indices": days.tolist(),
        }
    if len(set(all_days)) != len(all_days):
        raise AssertionError("Planning day splits overlap")
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
