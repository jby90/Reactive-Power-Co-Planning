"""Summarize model-mismatch outcomes with mismatch-replicate uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LEVEL_ORDER = (
    "none",
    "mild",
    "design",
    "moderate",
    "topology_a",
    "topology_b",
    "design_topology_a",
)


def cluster_bootstrap_interval(
    frame: pd.DataFrame,
    value: str,
    cluster: str = "mismatch_id",
    draws: int = 10000,
    seed: int = 20260919,
) -> tuple[float, float]:
    cluster_means = frame.groupby(cluster, sort=True)[value].mean().to_numpy(dtype=float)
    if cluster_means.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, cluster_means.size, size=(draws, cluster_means.size))
    estimates = cluster_means[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def summarize(root: Path, output: Path) -> pd.DataFrame:
    rows: list[dict] = []
    level_rows: list[dict] = []
    for level in LEVEL_ORDER:
        episodes_path = root / level / "episodes.csv"
        if not episodes_path.exists():
            raise FileNotFoundError(episodes_path)
        frame = pd.read_csv(episodes_path)
        for candidate_id, group in frame.groupby("candidate_id", sort=True):
            low, high = cluster_bootstrap_interval(group, "risk")
            rows.append(
                {
                    "level": level,
                    "candidate_id": int(candidate_id),
                    "days": int(group.scenario_index.nunique()),
                    "mismatch_replicates": int(group.mismatch_id.nunique()),
                    "episodes": int(len(group)),
                    "event_days": int(group.risk.sum()),
                    "event_rate": float(group.risk.mean()),
                    "cluster_bootstrap_95_low": low,
                    "cluster_bootstrap_95_high": high,
                    "mean_intervention_rate": float(group.intervention_rate.mean()),
                    "projection_failure_steps": int(group.projection_failures.sum()),
                    "mean_daily_loss_mwh": float(group.daily_loss_mwh.mean()),
                    "worst_actual_vmin": float(group.actual_vmin.min()),
                    "actual_vmin_p01": float(group.actual_vmin.quantile(0.01)),
                    "actual_vmin_p05": float(group.actual_vmin.quantile(0.05)),
                    "worst_actual_vmax": float(group.actual_vmax.max()),
                }
            )
        low, high = cluster_bootstrap_interval(frame, "risk")
        level_rows.append(
            {
                "level": level,
                "candidates": int(frame.candidate_id.nunique()),
                "days": int(frame.scenario_index.nunique()),
                "mismatch_replicates": int(frame.mismatch_id.nunique()),
                "episodes": int(len(frame)),
                "event_rate": float(frame.risk.mean()),
                "cluster_bootstrap_95_low": low,
                "cluster_bootstrap_95_high": high,
                "mean_intervention_rate": float(frame.intervention_rate.mean()),
                "projection_failure_steps": int(frame.projection_failures.sum()),
                "worst_actual_vmin": float(frame.actual_vmin.min()),
                "worst_actual_vmax": float(frame.actual_vmax.max()),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    candidate_summary = pd.DataFrame(rows)
    candidate_summary.to_csv(output / "candidate_summary.csv", index=False)
    pd.DataFrame(level_rows).to_csv(output / "level_summary.csv", index=False)
    metadata = {
        "uncertainty_unit": "mismatch replicate after averaging the 17 matched days",
        "bootstrap_draws": 10000,
        "bootstrap_seed": 20260919,
        "single_replicate_levels": ["none", "topology_a", "topology_b"],
        "single_replicate_interval": "not estimated",
    }
    (output / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return candidate_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    frame = summarize(args.root, args.out_dir)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
