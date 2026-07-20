"""Run and summarise the locked safety-projection confirmation study."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PYTHON = sys.executable
OUT_ROOT = ROOT / "reproduced" / "locked_controller_evaluation"
LOG_DIR = OUT_ROOT / "logs"
JOB_TABLES = {
    "uniform": ROOT / "data" / "results" / "locked_controller_evaluation" / "uniform1000.csv",
    "stress": ROOT / "data" / "results" / "locked_controller_evaluation" / "stress1000.csv",
}


def run_dir(family: str, seed: int) -> Path:
    if family == "wg":
        return ROOT / "models" / "wg_cvar" / f"seed{seed}"
    return ROOT / "models" / "concat" / f"seed{seed}"


def run_one(variant: str, seed: int, mode: str) -> None:
    out_dir = OUT_ROOT / variant / f"seed{seed}" / f"{mode}1000"
    if (out_dir / "summary.json").exists():
        return
    family, projected = variant.split("_")
    if projected == "projected":
        command = [
            PYTHON, str(SRC / "evaluate_safety_projected_policy.py"),
            "--run_dir", str(run_dir(family, seed)),
            "--jobs_csv", str(JOB_TABLES[mode]),
            "--workers", "4", "--out_dir", str(out_dir),
        ]
    else:
        command = [
            PYTHON, str(SRC / "evaluate_crdc_matched_jobs.py"),
            "--run_dir", str(run_dir(family, seed)),
            "--jobs_csv", str(JOB_TABLES[mode]),
            "--workers", "4", "--out_dir", str(out_dir),
        ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / f"{variant}_seed{seed}_{mode}.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def bootstrap_ci(values: np.ndarray, seed: int, samples: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, len(values), size=len(values))
        means[index] = float(values[draw].mean())
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def write_summary() -> None:
    variants = ("wg_projected", "concat_projected", "wg_raw", "concat_raw")
    rows = []
    for variant in variants:
        for seed in (42, 43, 44):
            for mode in ("uniform", "stress"):
                base = OUT_ROOT / variant / f"seed{seed}" / f"{mode}1000"
                summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
                rows.append({"variant": variant, "seed": seed, "mode": mode, **summary})
    summary_table = pd.DataFrame(rows)
    summary_table.to_csv(OUT_ROOT / "locked_summary.csv", index=False)

    comparisons = []
    per_mode_differences: dict[str, np.ndarray] = {}
    for mode in ("uniform", "stress"):
        seed_differences = []
        for seed in (42, 43, 44):
            proposed_path = OUT_ROOT / "wg_projected" / f"seed{seed}" / f"{mode}1000" / "episodes.csv"
            baseline_path = OUT_ROOT / "concat_projected" / f"seed{seed}" / f"{mode}1000" / "episodes.csv"
            proposed = pd.read_csv(proposed_path).sort_values("job_index")
            baseline = pd.read_csv(baseline_path).sort_values("job_index")
            if not np.array_equal(proposed.job_index.to_numpy(), baseline.job_index.to_numpy()):
                raise RuntimeError("Paired job indices do not match")
            difference = proposed.loss_mean_mw.to_numpy() - baseline.loss_mean_mw.to_numpy()
            seed_differences.append(difference)
            proposed_summary = summary_table[
                (summary_table.variant == "wg_projected")
                & (summary_table.seed == seed)
                & (summary_table["mode"] == mode)
            ].iloc[0]
            baseline_summary = summary_table[
                (summary_table.variant == "concat_projected")
                & (summary_table.seed == seed)
                & (summary_table["mode"] == mode)
            ].iloc[0]
            comparisons.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "wg_risk": proposed_summary.risk,
                    "concat_risk": baseline_summary.risk,
                    "wg_loss_mw": proposed_summary.mean_loss_mw,
                    "concat_loss_mw": baseline_summary.mean_loss_mw,
                    "loss_change_pct": 100.0
                    * (proposed_summary.mean_loss_mw / baseline_summary.mean_loss_mw - 1.0),
                    "wg_intervention_rate": proposed_summary.mean_intervention_rate,
                    "concat_intervention_rate": baseline_summary.mean_intervention_rate,
                }
            )
        per_mode_differences[mode] = np.vstack(seed_differences).mean(axis=0)
    comparison_table = pd.DataFrame(comparisons)
    comparison_table.to_csv(OUT_ROOT / "locked_paired_comparison.csv", index=False)

    mode_results = {}
    for offset, mode in enumerate(("uniform", "stress")):
        values = per_mode_differences[mode]
        lower, upper = bootstrap_ci(values, 2026071603 + offset)
        mode_results[mode] = {
            "mean_paired_loss_difference_mw": float(values.mean()),
            "bootstrap_95_ci_mw": [lower, upper],
        }
    combined = np.concatenate([per_mode_differences["uniform"], per_mode_differences["stress"]])
    combined_lower, combined_upper = bootstrap_ci(combined, 2026071605)

    proposed = summary_table[summary_table.variant == "wg_projected"]
    baseline = summary_table[summary_table.variant == "concat_projected"]
    risk_gate = True
    for seed in (42, 43, 44):
        for mode in ("uniform", "stress"):
            p = proposed[(proposed.seed == seed) & (proposed["mode"] == mode)].iloc[0]
            b = baseline[(baseline.seed == seed) & (baseline["mode"] == mode)].iloc[0]
            risk_gate = risk_gate and p.risk <= b.risk + 1e-12
    mean_loss_gate = all(
        proposed[proposed["mode"] == mode].mean_loss_mw.mean()
        < baseline[baseline["mode"] == mode].mean_loss_mw.mean()
        for mode in ("uniform", "stress")
    )
    no_pf = bool(summary_table.pf_fail_rate.max() == 0.0)
    intervention_gate = bool(proposed.mean_intervention_rate.max() <= 0.01)
    ci_gate = bool(combined_upper < 0.0)
    result = {
        "all_gates_pass": bool(risk_gate and mean_loss_gate and no_pf and intervention_gate and ci_gate),
        "risk_not_worse_every_seed_and_mode": bool(risk_gate),
        "mean_loss_lower_both_modes": bool(mean_loss_gate),
        "no_power_flow_failures": no_pf,
        "wg_intervention_rate_at_most_one_percent": intervention_gate,
        "combined_paired_bootstrap_ci_below_zero": ci_gate,
        "combined_mean_paired_loss_difference_mw": float(combined.mean()),
        "combined_bootstrap_95_ci_mw": [combined_lower, combined_upper],
        "mode_results": mode_results,
    }
    (OUT_ROOT / "locked_gate_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


def main() -> None:
    variants = ("wg_projected", "concat_projected", "wg_raw", "concat_raw")
    jobs = [
        (variant, seed, mode)
        for variant in variants
        for seed in (42, 43, 44)
        for mode in ("uniform", "stress")
    ]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_one, *job) for job in jobs]
        for future in futures:
            future.result()
    write_summary()


if __name__ == "__main__":
    main()
