"""Generate the locked confirmatory job tables without evaluating a policy."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPOSITORY_ROOT / "data" / "inputs"
OUT_DIR = REPOSITORY_ROOT / "reproduced" / "locked_controller_evaluation"
LOWER = np.array([0.7, 0.7, 0.0], dtype=np.float64)
UPPER = np.array([1.5, 1.5, 1.0], dtype=np.float64)
EPISODE_STEPS = 96
TOTAL_TIME_STEPS = int(np.load(DATA_DIR / "load96.npy").shape[0])


def make_jobs(seed: int, count: int, stress: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    max_start = max(1, TOTAL_TIME_STEPS - (EPISODE_STEPS + 1))
    corners = np.array(
        [[0.7, 0.7], [0.7, 1.5], [1.5, 0.7], [1.5, 1.5]], dtype=np.float64
    )
    for index in range(count):
        if stress and rng.random() >= 0.5:
            pv_scale, svc_scale = corners[int(rng.integers(0, len(corners)))]
            cap_total = rng.uniform(LOWER[2], UPPER[2])
            sample_kind = "corner"
        else:
            pv_scale, svc_scale, cap_total = rng.uniform(LOWER, UPPER)
            sample_kind = "uniform"
        rows.append(
            {
                "job_index": index,
                "pv_s_scale": float(np.float32(pv_scale)),
                "svc_q_scale": float(np.float32(svc_scale)),
                "cap_total_mvar": float(np.float32(cap_total)),
                "t0": int(rng.integers(0, max_start)),
                "sample_kind": sample_kind,
            }
        )
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specifications = {
        "uniform1000": {"seed": 2026071601, "stress": False},
        "stress1000": {"seed": 2026071602, "stress": True},
    }
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "capacity_lower": LOWER.tolist(),
        "capacity_upper": UPPER.tolist(),
        "episode_steps": EPISODE_STEPS,
        "total_time_steps": TOTAL_TIME_STEPS,
        "tables": {},
    }
    for name, specification in specifications.items():
        table = make_jobs(specification["seed"], 1000, specification["stress"])
        path = OUT_DIR / f"{name}.csv"
        table.to_csv(path, index=False)
        metadata["tables"][name] = {
            **specification,
            "rows": len(table),
            "path": str(path),
            "sha256": sha256(path),
        }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
