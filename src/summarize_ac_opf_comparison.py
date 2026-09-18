"""Summarise paired learned-control loss gaps to the local AC-OPF baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from statistics_utils import mean_t_interval


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    samples: int = 10000,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    batch = 500
    for start in range(0, samples, batch):
        stop = min(start + batch, samples)
        indices = rng.integers(0, len(finite), size=(stop - start, len(finite)))
        means[start:stop] = finite[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def compare_one(
    opf: pd.DataFrame,
    learned: pd.DataFrame,
    *,
    method: str,
    execution: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[dict]]:
    columns = [
        "candidate_id",
        "scenario_index",
        "daily_loss_mwh",
        "risk",
        "pf_fail",
    ]
    matched = learned[columns].merge(
        opf[columns],
        on=["candidate_id", "scenario_index"],
        suffixes=("_learned", "_opf"),
        validate="one_to_one",
    )
    if len(matched) != len(opf):
        raise ValueError(
            f"{method}/{execution} matched {len(matched)} of {len(opf)} AC-OPF days"
        )
    matched.insert(0, "execution", execution)
    matched.insert(0, "method", method)
    matched.insert(0, "seed", seed)
    matched["loss_gap_mwh"] = (
        matched.daily_loss_mwh_learned - matched.daily_loss_mwh_opf
    )
    matched["relative_loss_gap_percent"] = np.where(
        matched.daily_loss_mwh_opf > 0,
        100.0 * matched.loss_gap_mwh / matched.daily_loss_mwh_opf,
        np.nan,
    )

    rows = []
    for candidate_id, group in matched.groupby("candidate_id", sort=True):
        values = group.loss_gap_mwh.to_numpy(dtype=float)
        lower, upper = bootstrap_mean_ci(
            values,
            seed=2026091900
            + int(candidate_id)
            + (0 if method == "wg" else 1000)
            + (0 if execution == "raw" else 2000),
        )
        rows.append(
            {
                "method": method,
                "execution": execution,
                "seed": int(seed),
                "candidate_id": int(candidate_id),
                "matched_days": len(group),
                "mean_learned_loss_mwh": float(group.daily_loss_mwh_learned.mean()),
                "mean_opf_loss_mwh": float(group.daily_loss_mwh_opf.mean()),
                "mean_loss_gap_mwh": float(values.mean()),
                "median_loss_gap_mwh": float(np.median(values)),
                "bootstrap_95_ci_low_mwh": float(lower),
                "bootstrap_95_ci_high_mwh": float(upper),
                "mean_relative_loss_gap_percent": float(
                    group.relative_loss_gap_percent.mean()
                ),
                "fraction_learned_below_local_opf": float(np.mean(values < 0)),
                "learned_event_days": int(group.risk_learned.sum()),
                "opf_event_days": int(group.risk_opf.sum()),
                "learned_pf_fail_days": int(group.pf_fail_learned.sum()),
                "opf_pf_fail_days": int(group.pf_fail_opf.sum()),
            }
        )
    return matched, rows


def aggregate_seed_summaries(
    seed_summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate learned-minus-OPF gaps using training seed as the unit."""
    rows = []
    for (method, execution, candidate_id), group in seed_summary.groupby(
        ["method", "execution", "candidate_id"], sort=True
    ):
        seed_gaps = group.mean_loss_gap_mwh.to_numpy(dtype=float)
        gap_mean, gap_low, gap_high = mean_t_interval(seed_gaps)
        matched = paired[
            (paired.method == method)
            & (paired.execution == execution)
            & (paired.candidate_id == candidate_id)
        ]
        rows.append(
            {
                "method": method,
                "execution": execution,
                "candidate_id": int(candidate_id),
                "training_seeds": int(group.seed.nunique()),
                "matched_days_per_seed": int(group.matched_days.min()),
                "mean_learned_loss_mwh": float(group.mean_learned_loss_mwh.mean()),
                "mean_opf_loss_mwh": float(group.mean_opf_loss_mwh.mean()),
                "mean_loss_gap_mwh": gap_mean,
                "seed_t_95_ci_low_mwh": gap_low,
                "seed_t_95_ci_high_mwh": gap_high,
                "seed_gap_standard_deviation_mwh": float(seed_gaps.std(ddof=1)),
                "seed_mean_gaps_mwh": json.dumps(seed_gaps.tolist()),
                "mean_relative_loss_gap_percent": float(
                    group.mean_relative_loss_gap_percent.mean()
                ),
                "fraction_learned_below_local_opf": float(
                    (matched.loss_gap_mwh < 0).mean()
                ),
                "learned_event_days": int(matched.risk_learned.sum()),
                "opf_event_days": int(matched.risk_opf.sum()),
                "learned_pf_fail_days": int(matched.pf_fail_learned.sum()),
                "opf_pf_fail_days": int(matched.pf_fail_opf.sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opf_episodes", type=Path, required=True)
    parser.add_argument("--learned_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    opf = pd.read_csv(args.opf_episodes)
    all_matched = []
    summary_rows = []
    timing_rows = []
    timing_seed_rows = []
    for method in ("wg", "concat"):
        for seed in (42, 43, 44, 45, 46):
            for execution in ("raw", "projected"):
                directory = args.learned_root / f"{method}_seed{seed}_{execution}"
                if not directory.exists():
                    raise FileNotFoundError(directory)
                learned = pd.read_csv(directory / "episodes.csv")
                matched, rows = compare_one(
                    opf,
                    learned,
                    method=method,
                    execution=execution,
                    seed=seed,
                )
                all_matched.append(matched)
                summary_rows.extend(rows)
                learned_timing = json.loads(
                    (directory / "summary.json").read_text(encoding="utf-8")
                )
                timing_seed_rows.append(
                    {
                        "method": method,
                        "execution": execution,
                        "seed": seed,
                        "mean_online_ms": learned_timing["timing_ms"]["online_total"]["mean"],
                        "p95_online_ms": learned_timing["timing_ms"]["online_total"]["p95"],
                        "max_online_ms": learned_timing["timing_ms"]["online_total"]["max"],
                    }
                )
    timing_seed = pd.DataFrame(timing_seed_rows)
    for (method, execution), group in timing_seed.groupby(["method", "execution"]):
        timing_rows.append(
            {
                "method": method,
                "execution": execution,
                "training_seeds": int(group.seed.nunique()),
                "mean_online_ms": float(group.mean_online_ms.mean()),
                "p95_online_ms": float(group.p95_online_ms.max()),
                "max_online_ms": float(group.max_online_ms.max()),
            }
        )
    opf_steps = pd.read_csv(args.opf_episodes.parent / "steps.csv")
    converged = opf_steps[opf_steps.converged == 1]
    timing_rows.append(
        {
            "method": "local_ac_opf",
            "execution": "centralized_dispatch",
            "training_seeds": 0,
            "mean_online_ms": float(converged.solve_ms.mean()),
            "p95_online_ms": float(converged.solve_ms.quantile(0.95)),
            "max_online_ms": float(converged.solve_ms.max()),
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paired = pd.concat(all_matched, ignore_index=True)
    seed_summary = pd.DataFrame(summary_rows)
    summary = aggregate_seed_summaries(seed_summary, paired)
    timing = pd.DataFrame(timing_rows)
    paired.to_csv(args.out_dir / "paired_daily_comparison.csv", index=False)
    seed_summary.to_csv(args.out_dir / "seed_summary.csv", index=False)
    timing_seed.to_csv(args.out_dir / "timing_by_seed.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    timing.to_csv(args.out_dir / "timing.csv", index=False)
    report = {
        "paired_rows": len(paired),
        "comparisons": len(summary),
        "training_seeds": sorted(seed_summary.seed.unique().tolist()),
        "uncertainty_unit": "independently trained policy seed",
        "all_opf_days_feasible": bool(
            (opf.risk == 0).all() & (opf.pf_fail == 0).all()
        ),
        "interpretation": (
            "Loss gaps are learned minus a centralized local AC-OPF dispatch baseline; "
            "95% intervals use the five independently trained policy seeds. "
            "Because AC OPF is nonconvex, this comparator is not claimed to be a global lower bound."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
