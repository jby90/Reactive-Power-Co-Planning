"""Run the frozen VMOD physical/telemetry shift stress-test matrix."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from evaluate_vmod_operational_shift import replicate_event_rate_summary
from vmod_current_study import (
    EXTERNAL_PROFILES_33,
    EXTERNAL_PROTOCOL_33,
    main_run_dir,
    read_selected_margin,
)
from vmod_statistics import percentile_bootstrap_mean_interval


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
PROTOCOL = EXTERNAL_PROTOCOL_33
LEVELS = {
    "nominal": dict(replicates=1),
    "mild": dict(replicates=20, r_std=0.025, x_std=0.025, load_std=0.01, pv_std=0.01),
    "design": dict(replicates=20, r_std=0.05, x_std=0.05, load_std=0.02, pv_std=0.02),
    "moderate": dict(replicates=20, r_std=0.10, x_std=0.10, load_std=0.05, pv_std=0.05),
    "telemetry": dict(
        replicates=20,
        voltage_measurement_std=0.002,
        power_measurement_relative_std=0.02,
    ),
    "delay_1_step": dict(replicates=1, observation_delay_steps=1),
    "delay_4_steps": dict(replicates=1, observation_delay_steps=4),
    "topology_a": dict(replicates=1, topology="tie32_open18"),
    "topology_b": dict(replicates=1, topology="tie33_open11"),
    "design_topology_a": dict(
        replicates=20,
        r_std=0.05,
        x_std=0.05,
        load_std=0.02,
        pv_std=0.02,
        topology="tie32_open18",
    ),
}


def refresh_event_rate_intervals(level_out: Path, summary: dict) -> dict:
    """Rebuild bounded intervals from the immutable episode-level records."""
    episodes = pd.read_csv(level_out / "episodes.csv")
    seed_rates = (
        episodes.groupby("seed", sort=True).event.mean().to_numpy(dtype=float)
    )
    seed_mean, seed_low, seed_high = percentile_bootstrap_mean_interval(seed_rates)
    summary.update(
        {
            "interpretation": (
                "Empirical out-of-distribution stress test of a raw actor without "
                "online model use; not a formal robustness or safety guarantee."
            ),
            "seed_mean_event_rate": seed_mean,
            "seed_level_event_rate_ci95_low": seed_low,
            "seed_level_event_rate_ci95_high": seed_high,
            "seed_level_event_rate_interval_method": (
                "percentile bootstrap over independently trained actors; 10000 "
                "resamples; seed 20260921"
            ),
            **replicate_event_rate_summary(episodes),
        }
    )
    (level_out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary


def evaluate_level(
    level: str,
    config: dict,
    *,
    out: Path,
    theta: str,
    main_run: Path,
    environment: dict[str, str],
) -> dict:
    level_out = out / level
    if not (level_out / "summary.json").exists():
        command = [
            str(PYTHON),
            "-u",
            "evaluate_vmod_operational_shift.py",
            "--theta",
            theta,
            "--days_metadata",
            str(PROTOCOL / "selection_days.csv"),
            "--profile_dir",
            str(EXTERNAL_PROFILES_33),
            "--initialisation_dir",
            str(main_run / "policies"),
            "--shift_replicates",
            str(config.get("replicates", 1)),
            "--workers",
            "8",
            "--r_std",
            str(config.get("r_std", 0.0)),
            "--x_std",
            str(config.get("x_std", 0.0)),
            "--load_std",
            str(config.get("load_std", 0.0)),
            "--pv_std",
            str(config.get("pv_std", 0.0)),
            "--voltage_measurement_std",
            str(config.get("voltage_measurement_std", 0.0)),
            "--power_measurement_relative_std",
            str(config.get("power_measurement_relative_std", 0.0)),
            "--observation_delay_steps",
            str(config.get("observation_delay_steps", 0)),
            "--topology_reconfiguration",
            str(config.get("topology", "none")),
            "--svc_absorption_ratio",
            "1.0",
            "--out_dir",
            str(level_out),
        ]
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    summary = json.loads((level_out / "summary.json").read_text(encoding="utf-8"))
    summary = refresh_event_rate_intervals(level_out, summary)
    stochastic_replicates = int(summary["shift_replicates"]) > 1
    return {
        "level": level,
        "evaluated_shift_days": summary["evaluated_shift_days"],
        "event_days": summary["event_days"],
        "event_rate": summary["event_rate"],
        "seed_mean_event_rate": summary["seed_mean_event_rate"],
        "seed_level_event_rate_ci95_low": summary[
            "seed_level_event_rate_ci95_low"
        ],
        "seed_level_event_rate_ci95_high": summary[
            "seed_level_event_rate_ci95_high"
        ],
        "replicate_mean_event_rate": summary["replicate_mean_event_rate"],
        "replicate_level_event_rate_ci95_low": summary[
            "replicate_level_event_rate_ci95_low"
        ],
        "replicate_level_event_rate_ci95_high": summary[
            "replicate_level_event_rate_ci95_high"
        ],
        "uncertainty_unit": (
            "shift_replicate" if stochastic_replicates else "training_seed"
        ),
        "plot_mean_event_rate": (
            summary["replicate_mean_event_rate"]
            if stochastic_replicates
            else summary["seed_mean_event_rate"]
        ),
        "plot_event_rate_ci95_low": (
            summary["replicate_level_event_rate_ci95_low"]
            if stochastic_replicates
            else summary["seed_level_event_rate_ci95_low"]
        ),
        "plot_event_rate_ci95_high": (
            summary["replicate_level_event_rate_ci95_high"]
            if stochastic_replicates
            else summary["seed_level_event_rate_ci95_high"]
        ),
        "pf_fail_days": summary["pf_fail_days"],
        "mean_daily_line_loss_mwh": summary["mean_daily_line_loss_mwh"],
        "worst_minimum_voltage_pu": summary["worst_minimum_voltage_pu"],
        "worst_maximum_voltage_pu": summary["worst_maximum_voltage_pu"],
    }


def main() -> None:
    margin = read_selected_margin()
    main_run = main_run_dir(margin)
    prerequisites = (
        main_run / "confirmation_boundary_summary.json",
        main_run / "conditioning_ablation" / "summary.json",
        main_run / "final_baselines" / "summary.json",
        main_run / "conventional_capacity_boundaries" / "summary.json",
    )
    while not all(path.exists() for path in prerequisites):
        time.sleep(60)
    boundary = json.loads(prerequisites[0].read_text(encoding="utf-8"))
    if not boundary["boundary_confirmed"]:
        raise RuntimeError("The main capacity boundary did not pass confirmation")
    selected = next(
        point for point in boundary["points"] if point["role"] == "selected"
    )
    theta = ",".join(
        map(
            str,
            (
                selected["pv_s_scale"],
                selected["svc_q_scale"],
                selected["cap_total_mvar"],
            ),
        )
    )
    out = main_run / "operational_shift"
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"

    completed: dict[str, dict] = {}
    errors: dict[str, str] = {}
    # One shift level at a time avoids exhausting memory while the 69-bus
    # teacher is running; each level still parallelises its power flows.
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(
                evaluate_level,
                level,
                config,
                out=out,
                theta=theta,
                main_run=main_run,
                environment=environment,
            ): level
            for level, config in LEVELS.items()
        }
        for future in as_completed(futures):
            level = futures[future]
            try:
                completed[level] = future.result()
            except Exception as exc:
                errors[level] = f"{type(exc).__name__}: {exc}"
    if errors:
        out.mkdir(parents=True, exist_ok=True)
        (out / "failed_levels.json").write_text(
            json.dumps(errors, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"Operational-shift levels failed: {errors}")
    (out / "failed_levels.json").unlink(missing_ok=True)
    rows = [completed[level] for level in LEVELS]
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "level_summary.csv", index=False)
    aggregate = {
        "theta": theta,
        "capacity_label": selected["capacity_label"],
        "interpretation": (
            "Empirical raw-controller stress tests under pre-specified physical and "
            "telemetry shifts; not a formal guarantee."
        ),
        "levels": rows,
    }
    (out / "study_summary.json").write_text(
        json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
