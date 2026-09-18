"""Exact finite-sample and seed-aware audits for the study revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, t


def clopper_pearson_upper(events: int, trials: int, confidence: float = 0.95) -> float:
    """Return the exact one-sided binomial upper confidence limit."""
    if trials <= 0 or not 0 <= events <= trials:
        raise ValueError("Require 0 <= events <= trials and trials > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if events == trials:
        return 1.0
    return float(beta.ppf(confidence, events + 1, trials - events))


def zero_event_sample_size(target_probability: float, confidence: float = 0.95) -> int:
    """Smallest n for which 0/n has an exact upper limit below target."""
    if not 0.0 < target_probability < 1.0:
        raise ValueError("target_probability must lie strictly between zero and one")
    return int(np.floor(np.log(1.0 - confidence) / np.log(1.0 - target_probability)) + 1)


def mean_t_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and two-sided t interval, intended for independent training seeds."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float(values.mean()), float("nan"), float("nan")
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(values.size))
    half = float(t.ppf(0.5 + confidence / 2.0, values.size - 1) * sem)
    return mean, mean - half, mean + half


def audit_ablation(root: Path) -> dict:
    rows = []
    for variant in ("concat_raw", "concat_projected", "wg_raw", "wg_projected"):
        for seed in (42, 43, 44):
            path = root / "ablation" / variant / f"seed{seed}" / "episodes.csv"
            daily = pd.read_csv(path)
            for candidate_id, group in daily.groupby("candidate_id"):
                events = int(group["risk"].sum())
                trials = int(len(group))
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "candidate_id": int(candidate_id),
                        "events": events,
                        "trials": trials,
                        "risk_hat": events / trials,
                        "risk_upper_95": clopper_pearson_upper(events, trials),
                        "mean_daily_loss_mwh": float(group["daily_loss_mwh"].mean()),
                        "mean_intervention_rate": float(group["intervention_rate"].mean()),
                    }
                )
    table = pd.DataFrame(rows)

    seed_contrasts = []
    for candidate_id in sorted(table.candidate_id.unique()):
        pivot = table[table.candidate_id == candidate_id].pivot(
            index="seed", columns="variant", values="mean_daily_loss_mwh"
        )
        for suffix in ("raw", "projected"):
            contrast = (pivot[f"concat_{suffix}"] - pivot[f"wg_{suffix}"]).to_numpy()
            mean, low, high = mean_t_interval(contrast)
            seed_contrasts.append(
                {
                    "candidate_id": int(candidate_id),
                    "execution": suffix,
                    "n_training_seeds": int(contrast.size),
                    "concat_minus_wg_mean_mwh": mean,
                    "ci95_low_mwh": low,
                    "ci95_high_mwh": high,
                    "seed_values_mwh": contrast.tolist(),
                }
            )
    return {
        "risk_rows": rows,
        "seed_level_loss_contrasts": seed_contrasts,
        "zero_event_sample_sizes": {
            "upper_below_1_percent_at_95_percent_confidence": zero_event_sample_size(0.01, 0.95),
            "upper_below_0_5_percent_at_95_percent_confidence": zero_event_sample_size(0.005, 0.95),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs/capacity_planning/study_results"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/results/statistical_audit"),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    audit = audit_ablation(args.root)
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    pd.DataFrame(audit["risk_rows"]).to_csv(args.out / "risk_confidence.csv", index=False)
    pd.DataFrame(audit["seed_level_loss_contrasts"]).to_csv(
        args.out / "seed_level_loss_contrasts.csv", index=False
    )
    print(json.dumps(audit["zero_event_sample_sizes"], indent=2))


if __name__ == "__main__":
    main()
